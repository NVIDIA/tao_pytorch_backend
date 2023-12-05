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
import torch.nn as nn
import os
import shutil
import math
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from copy import deepcopy
from typing import Any
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
class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        """Init."""
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = lambda x: decay * (1 - math.exp(-float(x) / 2000))
        self.updates = 0
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        """Implementation of updating the module."""
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        """Update the EMA module."""
        self.updates += 1
        d = self.decay(self.updates)
        self._update(model, update_fn=lambda e, m: d * e + (1. - d) * m)

    def set(self, model):
        """Set the new EMA module."""
        self._update(model, update_fn=lambda e, m: m)


class OCDnetModel(pl.LightningModule):
    """PTL module for single stream OCDnet."""

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
        self.criterion = build_loss(self.experiment_config['train']['loss'])
        # init the model
        self._build_model(experiment_spec, export)
        self.model_ema = None
        if experiment_spec['train']['model_ema']:
            self.model_ema = ModelEmaV2(self.model, decay=experiment_spec['train']['model_ema_decay'])
            self.metrics.update({'ema_recall': 0, 'precision': 0, 'ema_hmean': 0, 'ema_best_model_epoch': 0})
        self.name = self.model.name
        if torch.cuda.device_count() > 1:
            self.experiment_config['distributed'] = True
        else:
            self.experiment_config['distributed'] = False

        self.train_loader = get_dataloader(self.train_dataset_config, self.experiment_config['distributed'])
        assert self.train_loader is not None, "Train loader does not exist."

        if 'validate_dataset' in self.experiment_config["dataset"]:
            self.validate_loader = get_dataloader(self.validate_dataset_config, False)
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

        if self.model_ema is not None:
            ema_preds = self.model_ema.module(batch['img'])
            boxes, scores = self.post_process(batch, ema_preds, is_output_polygon=self.metric_cls.is_output_polygon)
            ema_raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores), box_thresh=self.box_thresh)
            return (raw_metric, ema_raw_metric)
        return (raw_metric,)

    def validation_epoch_end(self, raw_metric):
        """Validation step end."""
        self.raw_metrics = []
        for p in all_gather(raw_metric):
            self.raw_metrics.extend(p)
        if self.model_ema is not None:
            ema_metrics = self.metric_cls.gather_measure([ema_metrics[1] for ema_metrics in self.raw_metrics])
            ema_recall = ema_metrics['recall'].avg
            ema_precision = ema_metrics['precision'].avg
            ema_hmean = ema_metrics['hmean'].avg
            self.log("ema_recall", ema_recall, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)
            self.log("ema_precision", ema_precision, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)
            self.log("ema_hmean", ema_hmean, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)

        metrics = self.metric_cls.gather_measure([metrics[0] for metrics in self.raw_metrics])
        recall = metrics['recall'].avg
        precision = metrics['precision'].avg
        hmean = metrics['hmean'].avg

        self.log("recall", recall, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)
        self.log("precision", precision, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)
        self.log("hmean", hmean, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)
        net_save_path_best = '{}/model_best.pth'.format(self.checkpoint_dir)
        if self.model_ema is not None:
            net_save_path_best_ema = '{}/model_best_ema.pth'.format(self.checkpoint_dir)

        save_best = False
        save_best_ema = False
        if self.validate_loader is not None and self.metric_cls is not None:
            if hmean >= self.metrics['hmean']:
                save_best = True
                self.metrics['train_loss'] = self.train_loss / self.train_loader_len
                self.metrics['hmean'] = hmean
                self.metrics['precision'] = precision
                self.metrics['recall'] = recall
                self.metrics['best_model_epoch'] = self.current_epoch

            if self.model_ema is not None:
                if ema_hmean >= self.metrics['ema_hmean']:
                    save_best_ema = True
                    self.metrics['ema_hmean'] = ema_hmean
                    self.metrics['ema_precision'] = ema_precision
                    self.metrics['ema_recall'] = ema_recall
                    self.metrics['ema_best_model_epoch'] = self.current_epoch
        else:
            if (self.train_loss / self.train_loader_len) <= self.metrics['train_loss']:
                save_best = True
                self.metrics['train_loss'] = self.train_loss / self.train_loader_len
                self.metrics['best_model_epoch'] = self.current_epoch
        best_str = 'current best, '
        for k, v in self.metrics.items():
            best_str += '{}: {:.6f}, '.format(k, v)
        self.print(best_str)

        self._save_checkpoint(self.current_epoch, net_save_path)
        if save_best:
            shutil.copy(net_save_path, net_save_path_best)
            self.print("Saving current best: {}".format(net_save_path_best))
        else:
            self.print("Saving checkpoint: {}".format(net_save_path))

        if save_best_ema:
            self._save_checkpoint(self.current_epoch, net_save_path_best_ema, save_ema=True)

        if self.trainer.is_global_zero:
            self.console_logger.info('**********************Start logging Evaluation Results **********************')
            self.console_logger.info('current_epoch : {}'.format(self.current_epoch))
            self.console_logger.info('lr : {:.9f}'.format(*self.scheduler.get_lr()))
            self.console_logger.info('recall : {:2.5f}'.format(recall))
            self.console_logger.info('precision : {:2.5f}'.format(precision))
            self.console_logger.info('hmean : {:2.5f}'.format(hmean))
            if self.model_ema:
                self.console_logger.info('ema_recall : {:2.5f}'.format(ema_recall))
                self.console_logger.info('ema_precision : {:2.5f}'.format(ema_precision))
                self.console_logger.info('ema_hmean : {:2.5f}'.format(ema_hmean))

        self.status_logging_dict["recall"] = str(recall)
        self.status_logging_dict["precision"] = str(precision)
        self.status_logging_dict["hmean"] = str(hmean)
        if self.model_ema:
            self.status_logging_dict["ema_recall"] = str(ema_recall)
            self.status_logging_dict["ema_precision"] = str(ema_precision)
            self.status_logging_dict["ema_hmean"] = str(ema_hmean)
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Evaluation metrics generated.",
            status_level=status_logging.Status.RUNNING
        )
        return metrics

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.experiment_config['train'][name]['type']
        module_args = self.experiment_config['train'][name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        if module_name == "SGD":
            module_args.pop("amsgrad")
        elif module_name == "Adam":
            module_args.pop("momentum")
        return getattr(module, module_name)(*args, **module_args)

    def _save_checkpoint(self, epoch, file_name, save_ema=False):
        """Saving checkpoints

        Args:
            epoch: Current epoch number
            log: The logging information of the epoch
            save_best: If True, rename the saved checkpoint with 'model_best' prefix
        """
        state_dict = self.model.state_dict()
        if save_ema:
            state_dict = self.model_ema.module.state_dict()
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

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """train batch end."""
        if self.model_ema is not None:
            self.model_ema.update(self.model)
