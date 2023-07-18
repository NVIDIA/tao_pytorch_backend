# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main PTL model file for Metric Learning Recognition."""

from collections import defaultdict
import os

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_metric_learning import losses, miners, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from nvidia_tao_pytorch.cv.metric_learning_recognition.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.metric_learning_recognition.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.re_identification.utils.scheduler import WarmupMultiStepLR
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


class MLRecogModel(pl.LightningModule):
    """PTL module for single stream Metric Learning Recognition. The training
    process is to minimize the distances between the embeddings of the same class
    and maximize the distances between the embeddings of different classes. The
    validation process is to evaluate the similarity search performance of the
    model on the validation reference and query datasets. The validation process
    only supports running on a single GPU.
    """

    def __init__(self, experiment_spec, results_dir, subtask="train"):
        """Initializes training for Metric Learning Recognition model.

        Args:
            experiment_spec (DictConfig): Configuration File
            results_dir (String): Path to save results
            subtask (String): The purpose of the model. Can be "train", "evaluate", "export", "inference" only
        """
        super().__init__()
        self.experiment_spec = experiment_spec
        self.results_dir = results_dir
        self.subtask = subtask

        self.status_logging_dict = {"train_loss": 0.0,
                                    "val Precision at Rank 1": 0.0}

        if subtask == "train":
            checkpoint = self.experiment_spec["train"]["resume_training_checkpoint_path"]
        elif subtask in ("evaluate", "export", "inference"):
            checkpoint = self.experiment_spec[subtask]["checkpoint"]
        if checkpoint:
            # Failure should always be caught before or after this warning
            if not os.path.exists(checkpoint):
                checkpoint_to_load = False
            else:
                checkpoint_to_load = True
        else:
            checkpoint_to_load = False
        self._build_model(experiment_spec, checkpoint_to_load=checkpoint_to_load)
        # Activates manual optimization
        self.automatic_optimization = False
        if self.subtask == "train":
            status_logging.get_status_logger().write(
                message="Preparing for training",
                status_level=status_logging.Status.RUNNING)
            self.my_loss_func = self.__make_loss(experiment_spec)
            (self.train_loader, self.query_loader, self.gallery_loader,
                self.dataset_dict) = build_dataloader(
                    cfg=self.experiment_spec,
                    mode="train")
            self.class_dict = self.dataset_dict["query"].class_dict
            self.load_tester()

    def load_tester(self):
        """Loads a `pytorch_metric_learning.testers.GlobalTwoStreamEmbeddingSpaceTester` to prepare for gallery-query similarity search evaluation."""
        # suppress end test print results
        def end_test_hook(tester):
            pass
        self.tester = testers.GlobalEmbeddingSpaceTester(
            batch_size=self.experiment_spec["train"]["val_batch_size"],
            end_of_testing_hook=end_test_hook,
            dataloader_num_workers=self.experiment_spec["dataset"]["workers"],
            accuracy_calculator=AccuracyCalculator(
                k="max_bin_count",
                return_per_class=self.experiment_spec[self.subtask]["report_accuracy_per_class"]),
        )

    def _build_model(self, experiment_spec, checkpoint_to_load=False):
        self.model = build_model(
            experiment_spec,
            checkpoint_to_load=checkpoint_to_load)
        if self.subtask != "train":
            self.model.eval()

    def train_dataloader(self):
        """Builds the dataloader for training.

        Returns:
            train_loader (torch.utils.data.Dataloader): Traininig Data.
        """
        return self.train_loader

    def configure_optimizers(self):
        """Configure optimizers for training.

        Returns:
            optim_dict1 (Dict[String, Object]): a map for trunk's optimizer, monitor and lr scheduler
            optim_dict2 ( Dict[String, Object]): a map for embedder's optimizer, monitor and lr scheduler
        """
        self.train_config = self.experiment_spec["train"]
        self.optim_config = self.train_config["optim"]
        optimizers = self.__make_optimizer()
        self.schedulers = {}
        for k, opt in optimizers.items():
            sch = WarmupMultiStepLR(
                opt, self.optim_config["steps"],
                gamma=self.optim_config["gamma"],
                warmup_factor=self.optim_config["warmup_factor"],
                warmup_iters=self.optim_config["warmup_iters"],
                warmup_method=self.optim_config["warmup_method"],
                last_epoch=self.current_epoch - 1,
            )
            self.schedulers[k] = sch
        optim_dict1 = {
            "optimizer": optimizers["trunk"],
            'monitor': None,
            "lr_scheduler": self.schedulers["trunk"]
        }
        optim_dict2 = {
            "optimizer": optimizers["embedder"],
            'monitor': None,
            "lr_scheduler": self.schedulers["embedder"]
        }
        return (optim_dict1, optim_dict2)

    def __make_module_optimizer(self, model_name):
        if model_name == "embedder":
            model = self.model.embedder
        elif model_name == "trunk":
            model = self.model.trunk
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.optim_config[model_name]["base_lr"]
            weight_decay = self.optim_config[model_name]["weight_decay"]
            if "bias" in key:
                lr = self.optim_config[model_name]["base_lr"] * self.optim_config[model_name]["bias_lr_factor"]
                weight_decay = self.optim_config[model_name]["weight_decay_bias"]
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        if self.optim_config["name"] == 'SGD':
            optimizer = getattr(torch.optim, self.optim_config["name"])(params, momentum=self.optim_config[model_name]["momentum"])
        else:
            optimizer = getattr(torch.optim, self.optim_config["name"])(params)
        return optimizer

    def __make_optimizer(self):
        embedder_optimizer = self.__make_module_optimizer("embedder")
        trunk_optimizer = self.__make_module_optimizer("trunk")
        optimizers = {
            "embedder": embedder_optimizer,
            "trunk": trunk_optimizer
        }
        return optimizers

    def training_step(self, batch):
        """Training step.

        Args:
            batch (torch.Tensor): Batch of data

        Returns:
            loss (torch.float32): Loss value for each step in training
        """
        data, labels = batch
        data = data.float()
        opt1, opt2 = self.optimizers()
        self.optimizer_dict = {'trunk': opt1, "embedder": opt2}
        self._zero_grad()
        outputs = self.model(data)
        loss = self.my_loss_func(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("trunk_base_lr", self.schedulers["trunk"].get_lr()[0],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("embedder_base_lr", self.schedulers["embedder"].get_lr()[0],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.manual_backward(loss)
        # clip gradients
        for opt in self.optimizer_dict.values():
            self.clip_gradients(
                opt,
                gradient_clip_val=self.experiment_spec['train']['clip_grad_norm'],
                gradient_clip_algorithm="norm"
            )
        self._step_optimizers()
        self.current_loss = loss

        return loss

    def get_query_accuracy(self):
        """Obtains the metric results of gallery-query similarity search evaluation.

        Returns:
            all_accuracies (Dict[str, float]): a map of default accuracy metrics from
                `pytorch_metric_learning.utils.accuracy_calculator.AccuracyCalculator`.
                Explanations see
                https://kevinmusgrave.github.io/pytorch-metric-learning/accuracy_calculation/#explanations-of-the-default-accuracy-metrics
        """
        if self.subtask == "train":
            epoch_log = self.current_epoch
        else:
            epoch_log = "eval mode"
        all_accuracies = self.tester.test(self.dataset_dict,
                                          epoch_log,  # used for logging
                                          self.model,  # your model
                                          splits_to_eval=[('query', ['gallery'])]
                                          )
        return all_accuracies

    def on_train_epoch_end(self):
        """Action on training epoch end."""
        self._step_schedulers()

    @rank_zero_only
    def update_validation_metrics(self):
        """Updates the validation metrics at the end of each validation epoch."""
        all_accuracies = self.get_query_accuracy()
        ami = all_accuracies['query']['AMI_level0']
        nmi = all_accuracies['query']['NMI_level0']
        mean_avg_prec = all_accuracies['query']['mean_average_precision_level0']
        mean_reciprocal_rank = all_accuracies['query']['mean_reciprocal_rank_level0']
        mean_r_precision = all_accuracies['query']['r_precision_level0']
        val_accuracy = all_accuracies['query']['precision_at_1_level0']

        self.status_logging_dict['val_AMI'] = ami
        self.status_logging_dict['val_NMI'] = nmi
        if self.experiment_spec['train']['report_accuracy_per_class']:
            self.status_logging_dict['val Mean Average Precision'] = sum(mean_avg_prec) / len(mean_avg_prec)
            self.status_logging_dict['val Mean Reciprocal Rank'] = sum(mean_reciprocal_rank) / len(mean_reciprocal_rank)
            self.status_logging_dict['val r-Precision'] = sum(mean_r_precision) / len(mean_r_precision)
            self.status_logging_dict['val Precision at Rank 1'] = sum(val_accuracy) / len(val_accuracy)

        else:
            self.status_logging_dict['val Mean Average Precision'] = mean_avg_prec
            self.status_logging_dict['val Mean Reciprocal Rank'] = mean_reciprocal_rank
            self.status_logging_dict['val r-Precision'] = mean_r_precision
            self.status_logging_dict['val Precision at Rank 1'] = val_accuracy

        # print out validation results
        print("============================================")
        print(f"Validation results at epoch {self.current_epoch}:")
        for k, v in self.status_logging_dict.items():
            if "val" in k:
                print(f"{k}: {v:.4f}")

        if self.experiment_spec['train']['report_accuracy_per_class']:
            print("\nValidation results per class:")
            for k, v in all_accuracies['query'].items():
                if "level" in k:
                    if isinstance(v, list):
                        print(f"  {k[:-7]}:")
                        for i, vv in enumerate(v):
                            print(f"    {self.class_dict[i]}: {vv:.4f}")
        print("============================================")

    def training_epoch_end(self, training_step_outputs):
        """Generates train and validation metrics and the end of training epoch.

        training_step_outputs (List[Dict[str, torch.Tensor]]): List of outputs from training_step.
        """
        average_train_loss = 0.0
        for out in training_step_outputs:
            average_train_loss += out['loss'].item()
        average_train_loss /= len(training_step_outputs)
        report_loss = np.around(average_train_loss, decimals=4)
        report_trunk_lr = '{:.4e}'.format(self.schedulers['trunk'].get_lr()[0])
        report_embedder_lr = '{:.4e}'.format(self.schedulers['embedder'].get_lr()[0])

        self.status_logging_dict['train_loss'] = report_loss
        self.status_logging_dict['trunk_base_lr'] = report_trunk_lr
        self.status_logging_dict['embedder_base_lr'] = report_embedder_lr

        # validation_epoch_end does not work here. Manually set it up.
        # add one to match the checkpoint saving epochs
        if (self.current_epoch + 1) % self.experiment_spec['train']['checkpoint_interval'] == 0:
            self.update_validation_metrics()

        # status loggings are rank zero only
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train and eval metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def _step_optimizers(self):
        for v in self.optimizer_dict.values():
            v.step()

    def _step_schedulers(self):
        for v in self.schedulers.values():
            v.step()

    def _zero_grad(self):
        self.model.zero_grad()
        for v in self.optimizer_dict.values():
            v.zero_grad()

    def on_train_epoch_start(self):
        """Perform on start of every epoch."""
        print('\n')

    def forward(self, x):
        """Forward of the Metric Learning Recognition model.

        Args:
            x (torch.Tensor): Batch of data

        Returns:
            output (torch.Tensor): Output of the model (class score, feats)
        """
        output = self.model(x)
        return output

    def __make_loss(self, cfg):
        self.optim_config = cfg["train"]["optim"]
        loss_func = losses.TripletMarginLoss(
            margin=self.optim_config["triplet_loss_margin"],
            smooth_loss=self.experiment_spec["train"]["smooth_loss"])
        mining_func = miners.MultiSimilarityMiner(
            epsilon=self.optim_config["miner_function_margin"])

        def calculate_loss(embeddings, labels):
            indices_tuple = mining_func(embeddings, labels)
            metric_loss = loss_func(embeddings, labels, indices_tuple)
            return metric_loss

        return calculate_loss

    def report_accuracies(self, acc_dict, save_results=False):
        """Converts the metrics results map to a pd.DataFrame table to display
        top1 precisions of all classes.

        Args:
            acc_dict (Dict[str, float]): A map of metrics and results obtained from
            `self.get_query_accuracies()`
            save_results (Boolean): If True, the derived dataframe would be saved
                to a csv file at output_dir/accuracy_per_class.csv

        Returns:
            df (pd.DataFrame): A table of top1 precision  of all classes.
        """
        output = defaultdict(dict)
        count = 0
        for idx in self.class_dict:
            class_name = self.class_dict[idx]
            if class_name in self.dataset_dict["query"].empty_classes:
                output[class_name]['top1_acc'] = None
                count += 1
            else:
                output[class_name]['top1_acc'] = \
                    acc_dict['query']['precision_at_1_level0'][idx - count]
        df = pd.DataFrame.from_dict(output, orient='index')
        if save_results:
            df.to_csv(os.path.join(
                self.results_dir,
                "accuracy_per_class.csv"))
        return df
