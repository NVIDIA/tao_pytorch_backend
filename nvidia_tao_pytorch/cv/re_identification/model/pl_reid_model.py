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

"""Main PTL model file for re-identification."""
from typing import Any, Dict
import pytorch_lightning as pl
import glob
import re
import os
import torch
import torch.nn.functional as F
import torchmetrics

from nvidia_tao_pytorch.cv.pose_classification.utils.common_utils import patch_decrypt_checkpoint
from nvidia_tao_pytorch.cv.re_identification.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.re_identification.model.losses.triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from nvidia_tao_pytorch.cv.re_identification.model.losses.center_loss import CenterLoss
from nvidia_tao_pytorch.cv.re_identification.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.re_identification.utils.reid_metric import R1_mAP, R1_mAP_reranking
from nvidia_tao_pytorch.cv.re_identification.lr_schedulers.warmup_multi_step_lr import WarmupMultiStepLR
from nvidia_tao_pytorch.cv.re_identification.lr_schedulers.cosine_lr import create_cosine_scheduler
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


# pylint:disable=too-many-ancestors
class ReIdentificationModel(pl.LightningModule):
    """PTL module for single stream re-identification."""

    def __init__(self, experiment_spec, prepare_for_training, export=False):
        """Initialize the ReIdentificationModel.

        Args:
            experiment_spec (DictConfig): Configuration File.
            prepare_for_training (bool): Boolean to set model based on training or testing/validation.
            export (bool, optional): Export model if True. Defaults to False.

        """
        super().__init__()
        self.experiment_spec = experiment_spec
        self.prepare_for_training = prepare_for_training
        # init the model
        self.model = self._build_model(experiment_spec, export)

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        if self.prepare_for_training:
            self.center_criterion = None
            if self.experiment_spec["model"]["with_center_loss"]:
                self.my_loss_func, self.center_criterion = self.__make_loss_with_center(experiment_spec, num_classes=self.num_classes)
            else:
                self.my_loss_func = self.__make_loss(experiment_spec, num_classes=self.num_classes)
            self.train_loader, self.val_loader, _, _ = build_dataloader(cfg=self.experiment_spec, is_train=True)

        self.status_logging_dict = {"train_loss": 0.0,
                                    "train_acc": 0.0,
                                    "cmc_rank_1": 0.0,
                                    "cmc_rank_5": 0.0,
                                    "cmc_rank_10": 0.0,
                                    "mAP": 0.0}

    def _build_model(self, experiment_spec, export):
        """Internal function to build the model.

        Args:
            experiment_spec (DictConfig): Configuration File.
            export (bool): Export model if True.

        Returns:
            model (Baseline): Model for re-identification.

        """
        if self.prepare_for_training:
            directory = experiment_spec["dataset"]["train_dataset_dir"]
            data = self.__process_dir(directory, relabel=True)
            self.num_classes, _, _ = self.__get_imagedata_info(data)
            self.query_dict = experiment_spec["dataset"]["query_dataset_dir"]
        else:
            self.num_classes = experiment_spec["dataset"]["num_classes"]
        self.model = build_model(experiment_spec, self.num_classes)
        return self.model

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader (Dataloader): Training Data.

        """
        return self.train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader (Dataloader): Validation Data.

        """
        return self.val_loader

    def configure_optimizers(self):
        """Configure optimizers for training.

        Returns:
            optim_dict (Dict): Optimizer Dictionary.

        """
        self.train_config = self.experiment_spec["train"]
        self.optim_config = self.train_config["optim"]
        optim_dict = {}

        optimizer, self.optimizer_center = self.__make_optimizer(self.center_criterion)

        if self.optim_config["warmup_method"] == "cosine":
            self.scheduler = create_cosine_scheduler(self.experiment_spec, optimizer)
        else:
            self.scheduler = WarmupMultiStepLR(optimizer, self.optim_config["lr_steps"],
                                               gamma=self.optim_config["gamma"],
                                               warmup_factor=self.optim_config["warmup_factor"],
                                               warmup_iters=self.optim_config["warmup_iters"],
                                               warmup_method=self.optim_config["warmup_method"])
            self.scheduler.step()
        optim_dict["optimizer"] = optimizer
        optim_dict["lr_scheduler"] = self.scheduler
        optim_dict['monitor'] = self.optim_config['lr_monitor']

        return optim_dict

    def __make_optimizer(self, center_criterion):
        """Make Optimizer.

        Returns:
            optimizer (Torch.Optimizer): Optimizer for training.

        """
        params = []
        for key, value in self.model.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.optim_config["base_lr"]
            weight_decay = self.optim_config["weight_decay"]
            if "bias" in key:
                lr = self.optim_config["base_lr"] * self.optim_config["bias_lr_factor"]
                weight_decay = self.optim_config["weight_decay_bias"]
            if self.optim_config["large_fc_lr"]:
                if "classifier" in key or "arcface" in key:
                    lr = self.optim_config["base_lr"] * 2
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        if self.optim_config["name"] == 'SGD':
            optimizer = getattr(torch.optim, self.optim_config["name"])(params, momentum=self.optim_config["momentum"])
        elif self.optim_config["name"] == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr=self.optim_config["base_lr"], weight_decay=self.optim_config["weight_decay"])
        else:
            optimizer = getattr(torch.optim, self.optim_config["name"])(params)

        optimizer_center = None
        if self.experiment_spec.model.with_center_loss:
            optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=self.optim_config["center_lr"])
        return optimizer, optimizer_center

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch (Tensor): Batch of data.
            batch_idx (int): Index of batch.

        Returns:
            loss (float): Loss value for each step in training.

        """
        data, label = batch
        data = data.float()
        if "swin" in self.experiment_spec.model.backbone:
            score, feat, _ = self.model(data)
        elif "resnet" in self.experiment_spec.model.backbone:
            score, feat = self.model(data)
        loss = self.my_loss_func(score, feat, label)
        self.train_accuracy.update(score, label)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log("base_lr", self.scheduler.get_lr()[0], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log("train_acc_1", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        """Log Training metrics to status.json"""
        average_train_loss = 0.0
        for out in training_step_outputs:
            average_train_loss += out['loss'].item()
        average_train_loss /= len(training_step_outputs)

        self.status_logging_dict["train_loss"] = average_train_loss
        self.status_logging_dict["train_acc"] = self.train_accuracy.compute().item()

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train and Val metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_validation_epoch_start(self):
        """Perform on validation."""
        if self.experiment_spec["re_ranking"]["re_ranking"]:
            self.metrics = R1_mAP_reranking(len(os.listdir(self.query_dict)), self.experiment_spec, self.prepare_for_training, feat_norm=True)
        else:
            self.metrics = R1_mAP(len(os.listdir(self.query_dict)), self.experiment_spec, self.prepare_for_training, feat_norm=True)
        self.metrics.reset()

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        data, pids, camids, img_path = batch
        if "swin" in self.experiment_spec.model.backbone:
            output, _ = self.model(data)
        elif "resnet" in self.experiment_spec.model.backbone:
            output = self.model(data)
        self.metrics.update(output, pids, camids, img_path)

    def on_validation_epoch_end(self):
        """Validation step end."""
        cmc, mAP = self.metrics.compute()
        for r in [1, 5, 10]:
            self.log(f"cmc_rank_{r}", cmc[r - 1], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
            self.status_logging_dict[f"cmc_rank_{r}"] = str(cmc[r - 1])
        self.log("mAP", mAP, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.status_logging_dict["mAP"] = str(mAP)

    def forward(self, x):
        """Forward of the re-identification model.

        Args:
            x (Tensor): Batch of data.

        Returns:
            output (Tensor): Output of the model (class score, feats).

        """
        output = self.model(x)
        return output

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Decrypt the checkpoint."""
        if checkpoint.get("state_dict_encrypted", False):
            # Retrieve encryption key from TLTPyTorchCookbook.
            key = TLTPyTorchCookbook.get_passphrase()
            if key is None:
                raise PermissionError("Cannot access model state dict without the encryption key")
            checkpoint = patch_decrypt_checkpoint(checkpoint, key)

    def __process_dir(self, dir_path, relabel=False):
        """Process the directory.

        Args:
            dir_path (str): Directory name.
            relabel (bool, optional): Enable relabelling feature if true, else disable. Defaults to False.

        Returns:
            dataset (Dataloader): Image data for training, testing, and validation.

        """
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            # assert 0 <= pid <= 1501, "The number of person IDs should be between 0 and 1501."
            # assert 1 <= camid <= 6, "The number of camera IDs should be between 0 and 6."
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def __get_imagedata_info(self, data):
        """Return meta data from the images.

        Args:
            data (Dataloader): Batch of data.

        Returns:
            num_pids (int): Number of person IDs.
            num_cams (int): Number of camera IDs.
            num_imgs (int): Number of images given a folder.

        """
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def __make_loss(self, cfg, num_classes):
        """Create a loss function based on the config.

        Args:
            cfg (DictConfig): Configuration File.
            num_classes (int): Number of classes.

        Returns:
            loss_func (Function Pointer): Loss function based on the config.

        """
        self.optim_config = cfg['train']["optim"]
        sampler = cfg['dataset']['sampler']
        if "triplet" in cfg['model']['metric_loss_type']:
            triplet = TripletLoss(self.optim_config["triplet_loss_margin"])  # triplet loss
        else:
            raise ValueError('Expected METRIC_LOSS_TYPE should be triplet'
                             'but got {}'.format(cfg['model']['metric_loss_type']))

        if cfg['model']['label_smooth']:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)

        if sampler == 'softmax':
            def loss_func(score, feat, target):
                return F.cross_entropy(score, target)
        elif cfg['dataset']['sampler'] == 'triplet':
            def loss_func(score, feat, target):
                return triplet(feat, target)[0]
        elif cfg['dataset']['sampler'] == 'softmax_triplet':
            def loss_func(score, feat, target):
                if 'triplet' in cfg['model']['metric_loss_type']:
                    if cfg['model']['label_smooth']:
                        return xent(score, target) + triplet(feat, target)[0]
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
                raise ValueError('Expected METRIC_LOSS_TYPE should be triplet'
                                 'but got {}'.format(cfg['model']['metric_loss_type']))
        else:
            raise ValueError('Expected sampler should be softmax, triplet or softmax_triplet, '
                             'but got {}'.format(cfg['dataset']['sampler']))
        return loss_func

    def __make_loss_with_center(self, cfg, num_classes):
        """Create a loss function with center loss based on the config.

        Args:
            cfg (DictConfig): Configuration File.
            num_classes (int): Number of classes.

        Returns:
            loss_func (Function Pointer): Loss function based on the config.

        """
        if cfg['model']['backbone'] == 'resnet18' or cfg['model']['backbone'] == 'resnet34':
            feat_dim = 512
        else:
            feat_dim = cfg['model']['feat_dim']

        if cfg['model']['metric_loss_type'] == 'center':
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)

        elif cfg['model']['metric_loss_type'] == 'triplet_center':
            triplet = TripletLoss(cfg['train']['optim']['triplet_loss_margin'])
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)

        else:
            raise ValueError('Expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                             'but got {}'.format(cfg['model']['metric_loss_type']))

        if cfg['model']['label_smooth']:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)

        def loss_func(score, feat, target):
            if cfg['model']['metric_loss_type'] == 'center':
                if cfg['model']['label_smooth']:
                    return xent(score, target) + \
                        self.optim_config['center_loss_weight'] * center_criterion(feat, target)
                return F.cross_entropy(score, target) + \
                    self.optim_config['center_loss_weight'] * center_criterion(feat, target)

            if cfg['model']['metric_loss_type'] == 'triplet_center':
                if cfg['model']['label_smooth']:
                    return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        self.optim_config['center_loss_weight'] * center_criterion(feat, target)
                return F.cross_entropy(score, target) + \
                    triplet(feat, target)[0] + \
                    self.optim_config['center_loss_weight'] * center_criterion(feat, target)

            raise ValueError('Expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                             'but got {}'.format(cfg['model']['metric_loss_type']))
        return loss_func, center_criterion
