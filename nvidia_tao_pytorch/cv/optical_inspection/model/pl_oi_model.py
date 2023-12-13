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

""" Main PTL model file for optical inspection"""
import math
import torch
from torch.nn import functional as F
import pandas as pd
import pytorch_lightning as pl
from typing import Optional, Dict, Any
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.optical_inspection.model.build_nn_model import (build_oi_model, ContrastiveLoss1, AOIMetrics)
from nvidia_tao_pytorch.cv.optical_inspection.utils.common_utils import patch_decrypt_checkpoint


class OpticalInspectionModel(pl.LightningModule):
    """Pytorch Lighting for Optical Inspection

    Args:
        experiment_config (OmegaConf.DictConf): The experiment configuration.
        export (bool): Flag indicating whether to export the model.
    """

    def __init__(self, experiment_spec, export=False, **kwargs):
        """Initialize"""
        super().__init__(**kwargs)
        self.experiment_spec = experiment_spec
        # init the model
        self._build_model(experiment_spec, export)
        self.tensorboard = experiment_spec.train.tensorboard
        self.status_logging_dict = {"train_loss": 0.0,
                                    "train_acc": 0.0,
                                    "train_fpr": 0.0,
                                    "val_loss": 0.0,
                                    "val_acc": 0.0,
                                    "val_fpr": 0.0}
        self.train_metrics = AOIMetrics()
        self.val_metrics = AOIMetrics()
        self.num_train_steps_per_epoch = None
        self.num_val_steps_per_epoch = None

    def _build_model(self, experiment_spec, export=False):
        self.model = build_oi_model(
            experiment_config=experiment_spec, export=export
        )
        print(self.model)

    def setup(self, stage: Optional[str] = None):
        """ Set up the dataset for train and val"""
        dataset_config = self.experiment_spec["dataset"]
        train_data_path = dataset_config["train_dataset"]["csv_path"]
        val_data_path = dataset_config["validation_dataset"]["csv_path"]
        self.df_train = pd.read_csv(train_data_path)
        self.df_valid = pd.read_csv(val_data_path)

    def train_dataloader(self):
        """Build the dataloader for training."""
        train_loader = build_dataloader(df=self.df_train,
                                        weightedsampling=True,
                                        split='train',
                                        data_config=self.experiment_spec["dataset"])
        self.num_train_steps_per_epoch = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
        logging.info("Number of steps for training: {}".format(self.num_train_steps_per_epoch))
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for training."""
        val_loader = build_dataloader(df=self.df_valid,
                                      weightedsampling=False,
                                      split='valid',
                                      data_config=self.experiment_spec["dataset"])
        self.num_val_steps_per_epoch = math.ceil(len(val_loader.dataset) / val_loader.batch_size)
        logging.info("Number of steps for validation: {}".format(self.num_val_steps_per_epoch))
        return val_loader

    def configure_optimizers(self):
        """Configure optimizers for training"""
        optim_dict = {}
        train_config = self.experiment_spec["train"]
        if train_config['optim']['type'] == 'Adam':
            optim = torch.optim.Adam(self.parameters(), train_config['optim']['lr'])
        else:
            optim = torch.optim.SGD(params=self.parameters(),
                                    lr=train_config['optim']['lr'],
                                    momentum=train_config['optim']['momentum'],
                                    weight_decay=train_config['optim']['weight_decay'])

        optim_dict["optimizer"] = optim
        return optim_dict

    def training_step(self, batch, batch_idx):
        """Perform a single training step.

        Args:
            batch: The input batch containing img0, img1, and label tensors.
            batch_idx: Index of the current batch.

        Returns:
            The computed loss value for the training step.
        """
        margin = self.experiment_spec["model"]["margin"]
        img0, img1, label = batch
        self.visualize_image(
            "compare_sample", img0,
            logging_frequency=self.tensorboard.infrequent_logging_frequency
        )
        self.visualize_image(
            "golden_sample", img1,
            logging_frequency=self.tensorboard.infrequent_logging_frequency
        )
        criterion = ContrastiveLoss1(margin)
        output1, output2 = self.model(img0, img1)
        loss = criterion(output1, output2, label)
        euclidean_distance = F.pairwise_distance(output1, output2)
        self.train_metrics.update(euclidean_distance, label)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

    def visualize_histogram(self, logging_frequency=2):
        """Visualize histograms of model parameters.

        Args:
            logging_frequency (int): The frequency at which to log the histograms.
        """
        if self.current_epoch % logging_frequency == 0 and self.tensorboard.enabled:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(f"histogram/{name}", params, self.current_epoch)

    def visualize_image(self, name, tensor, logging_frequency=10):
        """Visualize images during training.

        Args:
            name (str): The name for the image to be visualized.
            tensor (torch.Tensor): The input tensor containing the image.
            logging_frequency (int): The frequency at which to log the images.

        """
        logging_frequency_in_steps = self.num_train_steps_per_epoch * logging_frequency
        is_log_step = self.global_step % logging_frequency_in_steps == 0
        if is_log_step and self.tensorboard.enabled:
            self.logger.experiment.add_images(
                f"image/{name}",
                tensor,
                global_step=self.global_step
            )

    def visualize_metrics(self, metrics_dict):
        """Visualize metrics from during train/val.

        Args:
            metrics_dict (dict): Dictionary of metric tensors to be visualized.

        Returns:
            No returns.
        """
        # Make sure the scalars are logged in TensorBoard always.
        if self.logger:
            self.logger.log_metrics(metrics_dict, step=self.current_epoch)

    def training_epoch_end(self, training_step_outputs):
        """Log Training metrics to status.json

        Args:
            training_step_outputs: List of outputs from training steps in the epoch.
        """
        average_train_loss = 0.0
        for out in training_step_outputs:
            if isinstance(out, tuple):
                average_train_loss += out[0].item()
            else:
                average_train_loss += out['loss'].item()
        average_train_loss /= len(training_step_outputs)

        train_accuracy = self.train_metrics.compute()['total_accuracy'].item()
        train_false_positive_rate = self.train_metrics.compute()['false_alarm'].item()
        self.status_logging_dict["train_loss"] = average_train_loss
        self.status_logging_dict["train_acc"] = train_accuracy
        self.status_logging_dict["train_fpr"] = train_false_positive_rate
        tensorboard_logging_dict = {
            "train_acc": train_accuracy,
            "train_fpr": train_false_positive_rate
        }
        self.visualize_histogram(
            logging_frequency=self.tensorboard.infrequent_logging_frequency
        )
        self.visualize_metrics(tensorboard_logging_dict)
        self.train_metrics.reset()

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train and Val metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch: The input batch containing img0, img1, and label tensors.
            batch_idx: Index of the current batch.

        Returns:
            loss: Validation loss value.
        """
        margin = self.experiment_spec["model"]["margin"]
        img0, img1, label = batch
        criterion = ContrastiveLoss1(margin)
        output1, output2 = self.model(img0, img1)
        loss = criterion(output1, output2, label)
        euclidean_distance = F.pairwise_distance(output1, output2)

        self.val_metrics.update(euclidean_distance, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        """Log Validation metrics to status.json

        Args:
            validation_step_outputs: List of outputs from validation steps in the epoch.
        """
        average_val_loss = 0.0
        for out in validation_step_outputs:
            if isinstance(out, tuple):
                average_val_loss += out[0].item()
            else:
                average_val_loss += out.item()

        average_val_loss /= len(validation_step_outputs)

        val_accuracy = self.val_metrics.compute()['total_accuracy'].item()
        val_fpr = self.val_metrics.compute()['false_alarm'].item()
        self.status_logging_dict["val_loss"] = average_val_loss
        self.status_logging_dict["val_acc"] = val_accuracy
        self.status_logging_dict["val_fpr"] = val_fpr
        validation_logging_dict = {
            "val_acc": val_accuracy,
            "val_fpr": val_fpr
        }
        self.visualize_metrics(validation_logging_dict)
        self.val_metrics.reset()

    def forward(self, x):
        """Forward of the Optical Inspection model.

        Args:
            x (torch.Tensor): Input data containing two images.

        Returns:
            output (torch.Tensor): Output of the model.
        """
        x1 = x[0]
        x2 = x[1]
        output = self.model(x1, x2)
        return output

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Encrpyt the checkpoint. The encryption is done in TLTCheckpointConnector."""
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Decrpyt the checkpoint"""
        if checkpoint.get("state_dict_encrypted", False):
            # Retrieve encryption key from TLTPyTorchCookbook.
            key = TLTPyTorchCookbook.get_passphrase()
            if key is None:
                raise PermissionError("Cannot access model state dict without the encryption key")
            checkpoint = patch_decrypt_checkpoint(checkpoint, key)

    def load_final_model(self) -> None:
        """Loading a pre-trained network weights"""
        model_path = self.experiment_spec['inference']['checkpoint']
        gpu_device = self.experiment_spec['inference']['gpu_id']
        siamese_inf = self.model.cuda().load_state_dict(torch.load(model_path, map_location='cuda:' + str(gpu_device)))
        return siamese_inf
