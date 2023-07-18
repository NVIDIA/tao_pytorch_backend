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

"""Main PTL model file for OCDnet."""

import torch
import os
import shutil

import pytorch_lightning as pl
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import all_gather
from nvidia_tao_pytorch.cv.ocdnet.model.build_nn_model import build_ocd_model
from nvidia_tao_pytorch.cv.ocdnet.data_loader.build_dataloader import get_dataloader
from nvidia_tao_pytorch.cv.ocdnet.lr_schedulers.schedulers import WarmupPolyLR
from nvidia_tao_pytorch.cv.ocdnet.model.model import build_loss
from nvidia_tao_pytorch.cv.ocdnet.post_processing.seg_detector_representer import get_post_processing
from nvidia_tao_pytorch.cv.ocdnet.utils.ocr_metric.icdar2015.quad_metric import get_metric
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.ocdnet.utils.util import create_logger


# pylint:disable=too-many-ancestors
class OCDnetModel(pl.LightningModule):
    """PTL module for OCDnet model."""

    def __init__(self, experiment_spec, export=False):
        """Init training for OCDnet model.

        Args:
            experiment_spec (dict): The experiment specification.
            export (bool, optional): Whether to build the model that can be exported to ONNX format. Defaults to False
        """
        super().__init__()
        self.experiment_config = experiment_spec
        self.train_dataset_config = experiment_spec["dataset"]["train_dataset"]
        self.validate_dataset_config = experiment_spec["dataset"]["validate_dataset"]
        self.model_config = experiment_spec["model"]
        self.train_config = experiment_spec["train"]
        self.epochs = self.train_config["num_epochs"]
        self.post_process = get_post_processing(self.train_config['post_processing'])
        self.box_thresh = self.train_config['post_processing']["args"]["box_thresh"]
        self.checkpoint_dir = self.experiment_config["train"]["results_dir"]
        self.metrics = {'recall': 0, 'precision': 0, 'hmean': 0, 'train_loss': float('inf'), 'best_model_epoch': 0}
        self.train_loss = 0.0

        # init the model
        self._build_model(experiment_spec, export)
        self.name = self.model.name

        if torch.cuda.device_count() > 1:
            self.experiment_config['distributed'] = True
        else:
            self.experiment_config['distributed'] = False

        self.train_loader = get_dataloader(self.experiment_config["dataset"]['train_dataset'], self.experiment_config['distributed'])
        assert self.train_loader is not None, "Train loader does not exist."

        if 'validate_dataset' in self.experiment_config["dataset"]:
            self.validate_loader = get_dataloader(self.experiment_config["dataset"]['validate_dataset'], False)
        else:
            self.validate_loader = None

        self.train_loader_len = len(self.train_loader)

        self.console_logger = create_logger()
        self.status_logging_dict = {}

    def _build_model(self, experiment_spec, export):
        """Internal function to build the model.

        This method constructs a model using the specified experiment specification and export flag. It returns the model.

        Args:
            experiment_spec (dict): The experiment specification.
            export (bool): Whether to build the model that can be exported to ONNX format.
        """
        self.model = build_ocd_model(experiment_config=experiment_spec,
                                     export=export)

    def forward(self, x):
        """Forward of the ocdnet model."""
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch (Tensor): Batch of data.
            batch_idx (int): Index of batch.

        Returns:
            loss (float): Loss value for each step in training.

        """
        self.train_loss = 0.

        preds = self.model(batch['img'])
        self.criterion = build_loss(self.experiment_config['train']['loss']).cuda()
        loss_dict = self.criterion(preds, batch)

        loss = loss_dict['loss']
        self.train_loss += loss

        return loss

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader (Dataloader): Traininig Data.

        """
        return self.train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader (Dataloader): Validation Data.

        """
        return self.validate_loader

    def configure_optimizers(self):
        """Configure optimizers for training"""
        optim_dict = {}

        self.warmup_epochs = self.experiment_config['train']['lr_scheduler']['args']['warmup_epoch']
        self.warmup_iters = self.warmup_epochs * self.train_loader_len

        self.optimizer = self._initialize('optimizer', torch.optim, self.model.parameters())

        self.scheduler = WarmupPolyLR(self.optimizer, max_iters=self.epochs * self.train_loader_len,
                                      warmup_iters=self.warmup_iters, warmup_epochs=self.warmup_epochs, epochs=self.epochs,
                                      **self.experiment_config['train']['lr_scheduler']['args'])

        optim_dict["optimizer"] = self.optimizer
        optim_dict["lr_scheduler"] = self.scheduler

        return optim_dict

    def on_train_epoch_start(self):
        """Perform on start of every epoch."""
        print('\n')

    def on_validation_epoch_start(self):
        """Perform on validation."""
        self.raw_metrics = []

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        preds = self.model(batch['img'])
        self.metric_cls = get_metric(self.experiment_config['train']['metric'])
        boxes, scores = self.post_process(batch, preds, is_output_polygon=self.metric_cls.is_output_polygon)
        raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores), box_thresh=self.box_thresh)
        return raw_metric

    def validation_epoch_end(self, raw_metric):
        """Validation step end."""
        print('\n')
        self.raw_metrics = []
        for p in all_gather(raw_metric):
            self.raw_metrics.extend(p)
        metrics = self.metric_cls.gather_measure(self.raw_metrics)
        self.log("recall", metrics['recall'].avg, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)
        self.log("precision", metrics['precision'].avg, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)
        self.log("hmean", metrics['hmean'].avg, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)

        try:
            os.makedirs(self.checkpoint_dir)
        except OSError:
            pass

        net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)
        net_save_path_best = '{}/model_best.pth'.format(self.checkpoint_dir)

        self._save_checkpoint(self.current_epoch, net_save_path)

        save_best = False
        if self.validate_loader is not None and self.metric_cls is not None:
            recall = metrics['recall'].avg
            precision = metrics['precision'].avg
            hmean = metrics['hmean'].avg

            if hmean >= self.metrics['hmean']:
                save_best = True
                self.metrics['train_loss'] = self.train_loss / self.train_loader_len
                self.metrics['hmean'] = hmean
                self.metrics['precision'] = precision
                self.metrics['recall'] = recall
                self.metrics['best_model_epoch'] = self.current_epoch
        else:
            if (self.train_loss / self.train_loader_len) <= self.metrics['train_loss']:
                save_best = True
                self.metrics['train_loss'] = self.train_loss / self.train_loader_len
                self.metrics['best_model_epoch'] = self.current_epoch
        best_str = 'current best, '
        for k, v in self.metrics.items():
            best_str += '{}: {:.6f}, '.format(k, v)
        self.print(best_str)
        if save_best:
            shutil.copy(net_save_path, net_save_path_best)
            self.print("Saving current best: {}".format(net_save_path_best))
        else:
            self.print("Saving checkpoint: {}".format(net_save_path))

        if self.trainer.is_global_zero:
            self.console_logger.info('**********************Start logging Evaluation Results **********************')
            self.console_logger.info('current_epoch : {}'.format(self.current_epoch))
            self.console_logger.info('lr : {:.9f}'.format(*self.scheduler.get_lr()))
            self.console_logger.info('recall : {:2.5f}'.format(recall))
            self.console_logger.info('precision : {:2.5f}'.format(precision))
            self.console_logger.info('hmean : {:2.5f}'.format(hmean))

        self.status_logging_dict["recall"] = str(recall)
        self.status_logging_dict["precision"] = str(precision)
        self.status_logging_dict["hmean"] = str(hmean)
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Evaluation metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.experiment_config['train'][name]['type']
        module_args = self.experiment_config['train'][name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def _save_checkpoint(self, epoch, file_name):
        """Saving checkpoints

        Args:
            epoch: Current epoch number
            log: The logging information of the epoch
            save_best: If True, rename the saved checkpoint with 'model_best' prefix
        """
        state_dict = self.model.state_dict()

        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.experiment_config,
            'metrics': self.metrics
        }

        torch.save(state, file_name)
