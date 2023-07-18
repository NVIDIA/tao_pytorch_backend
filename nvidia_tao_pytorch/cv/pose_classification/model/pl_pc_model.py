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

"""Main PTL model file for pose classification."""

from typing import Any, Dict
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import torchmetrics

from nvidia_tao_pytorch.cv.pose_classification.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.pose_classification.model.build_nn_model import build_pc_model
from nvidia_tao_pytorch.cv.pose_classification.utils.common_utils import patch_decrypt_checkpoint
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


# pylint:disable=too-many-ancestors
class PoseClassificationModel(pl.LightningModule):
    """PyTorch Lightning module for single stream pose classification."""

    def __init__(self, experiment_spec, export=False):
        """
        Initialize the training for pose classification model.

        Args:
            experiment_spec (dict): The experiment specifications.
            export (bool, optional): If set to True, the model is prepared for export. Defaults to False.
        """
        super().__init__()
        self.experiment_spec = experiment_spec
        self.model_config = experiment_spec["model"]
        self.dataset_config = experiment_spec["dataset"]

        # init the model
        self._build_model(experiment_spec, export)

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.status_logging_dict = {"train_loss": 0.0,
                                    "train_acc": 0.0,
                                    "val_loss": 0.0,
                                    "val_acc": 0.0}

    def _build_model(self, experiment_spec, export):
        """
        Internal function to build the model.

        Args:
            experiment_spec (dict): The experiment specifications.
            export (bool): If set to True, the model is prepared for export.
        """
        self.model = build_pc_model(experiment_config=experiment_spec,
                                    export=export)
        print(self.model)

    def train_dataloader(self):
        """
        Build the dataloader for training.

        Returns:
            DataLoader: The dataloader for training.
        """
        train_loader = \
            build_dataloader(data_path=self.dataset_config["train_dataset"]["data_path"],
                             label_path=self.dataset_config["train_dataset"]["label_path"],
                             label_map=self.dataset_config["label_map"],
                             random_choose=self.dataset_config["random_choose"],
                             random_move=self.dataset_config["random_move"],
                             window_size=self.dataset_config["window_size"],
                             mmap=True,
                             batch_size=self.dataset_config["batch_size"],
                             shuffle=False,
                             num_workers=self.dataset_config["num_workers"],
                             pin_mem=False)
        return train_loader

    def val_dataloader(self):
        """
        Build the dataloader for validation.

        Returns:
            DataLoader: The dataloader for validation.
        """
        val_loader = \
            build_dataloader(data_path=self.dataset_config["val_dataset"]["data_path"],
                             label_path=self.dataset_config["val_dataset"]["label_path"],
                             label_map=self.dataset_config["label_map"],
                             batch_size=self.dataset_config["batch_size"],
                             num_workers=self.dataset_config["num_workers"])
        return val_loader

    def configure_optimizers(self):
        """
        Configure optimizers for training.

        Returns:
            dict: A dictionary containing the optimizer, learning rate scheduler, and the metric to monitor.

        Raises:
            NotImplementedError: If an unsupported scheduler type is provided.
        """
        self.train_config = self.experiment_spec["train"]
        optim_dict = {}
        optim = torch.optim.SGD(params=self.parameters(),
                                lr=self.train_config['optim']['lr'],
                                momentum=self.train_config['optim']['momentum'],
                                nesterov=self.train_config['optim']['nesterov'],
                                weight_decay=self.train_config['optim']['weight_decay'])
        optim_dict["optimizer"] = optim
        scheduler_type = self.train_config['optim']['lr_scheduler']
        if scheduler_type == "AutoReduce":
            lr_scheduler = ReduceLROnPlateau(optim, 'min',
                                             patience=self.train_config['optim']['patience'],
                                             min_lr=self.train_config['optim']['min_lr'],
                                             factor=self.train_config['optim']["lr_decay"],
                                             verbose=True)
        elif scheduler_type == "MultiStep":
            lr_scheduler = MultiStepLR(optimizer=optim,
                                       milestones=self.train_config['optim']["lr_steps"],
                                       gamma=self.train_config['optim']["lr_decay"],
                                       verbose=True)
        else:
            raise NotImplementedError("Only [AutoReduce, MultiStep] schedulers are supported")

        optim_dict["lr_scheduler"] = lr_scheduler
        optim_dict['monitor'] = self.train_config['optim']['lr_monitor']

        return optim_dict

    def training_step(self, batch, batch_idx):
        """
        Define a training step.

        Args:
            batch (Tensor): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The computed loss.
        """
        data, label = batch
        output = self.model(data)
        loss = F.cross_entropy(output, label)
        self.train_accuracy.update(output, label)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc@1", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def training_epoch_end(self, training_step_outputs):
        """
        Log Training metrics to status.json at the end of the epoch.

        Args:
            training_step_outputs (list): List of outputs from each training step.
        """
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

    def validation_step(self, batch, batch_idx):
        """
        Define a validation step.

        Args:
            batch (Tensor): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The computed loss.
        """
        data, label = batch
        output = self.model(data)
        loss = F.cross_entropy(output, label)
        self.val_accuracy.update(output, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc@1", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        """
        Log Validation metrics to status.json at the end of the epoch.

        Args:
            validation_step_outputs (list): List of outputs from each validation step.
        """
        average_val_loss = 0.0
        for out in validation_step_outputs:
            average_val_loss += out.item()
        average_val_loss /= len(validation_step_outputs)

        self.status_logging_dict["val_loss"] = average_val_loss
        self.status_logging_dict["val_acc"] = self.val_accuracy.compute().item()

    def forward(self, x):
        """
        Define the forward pass of the pose classification model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        output = self.model(x)
        return output

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Encrypt the checkpoint. The encryption is done in TLTCheckpointConnector.

        Args:
            checkpoint (dict): The checkpoint to save.
        """
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Decrypt the checkpoint.

        Args:
            checkpoint (dict): The checkpoint to load.

        Raises:
            PermissionError: If the checkpoint is encrypted and the encryption key is not available.
        """
        if checkpoint.get("state_dict_encrypted", False):
            # Retrieve encryption key from TLTPyTorchCookbook.
            key = TLTPyTorchCookbook.get_passphrase()
            if key is None:
                raise PermissionError("Cannot access model state dict without the encryption key")
            checkpoint = patch_decrypt_checkpoint(checkpoint, key)
