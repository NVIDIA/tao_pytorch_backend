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

""" Main PTL model file for action recognition """

from typing import Any, Dict, Optional
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import torchmetrics

from nvidia_tao_pytorch.cv.action_recognition.dataloader.build_data_loader import build_dataloader, list_dataset
from nvidia_tao_pytorch.cv.action_recognition.model.build_nn_model import build_ar_model
from nvidia_tao_pytorch.cv.action_recognition.utils.common_utils import patch_decrypt_checkpoint
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


# pylint:disable=too-many-ancestors
class ActionRecognitionModel(pl.LightningModule):
    """ PTL module for action recognition model."""

    def __init__(self, experiment_spec, export=False):
        """Init training for 2D/3D action recognition model.

        Args:
            experiment_spec (dict): The experiment specification.
            export (bool, optional): Whether to build the model that can be exported to ONNX format. Defaults to False.
        """
        super().__init__()
        self.experiment_spec = experiment_spec
        self.dataset_config = experiment_spec["dataset"]
        self.model_config = experiment_spec["model"]
        self.data_shape = [self.model_config.input_height, self.model_config.input_width]

        # init the model
        self._build_model(experiment_spec, export)

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.status_logging_dict = {"train_loss": 0.0,
                                    "train_acc": 0.0,
                                    "val_loss": 0.0,
                                    "val_acc": 0.0}

    def _build_model(self, experiment_spec, export):
        """Internal function to build the model.

        This method constructs a model using the specified experiment specification and export flag. It returns the model.

        Args:
            experiment_spec (dict): The experiment specification.
            export (bool): Whether to build the model that can be exported to ONNX format.
        """
        self.model = build_ar_model(experiment_config=experiment_spec,
                                    export=export)
        print(self.model)

    def setup(self, stage: Optional[str] = None):
        """ Set up the dataset for train and val"""
        train_top_dir = self.dataset_config["train_dataset_dir"]
        val_top_dir = self.dataset_config["val_dataset_dir"]
        if train_top_dir is not None:
            self.train_dict = list_dataset(train_top_dir)
        else:
            raise ValueError("Please set the train dataset in the spec file")

        if val_top_dir is not None:
            self.val_dict = list_dataset(val_top_dir)
        else:
            self.val_dict = {}

        print("Train dataset samples: {}".format(len(self.train_dict)))
        print("Validation dataset samples: {}".format(len(self.val_dict)))

    def train_dataloader(self):
        """Build the dataloader for training."""
        train_loader = \
            build_dataloader(sample_dict=self.train_dict,
                             model_config=self.model_config,
                             output_shape=self.data_shape,
                             label_map=self.dataset_config["label_map"],
                             dataset_mode="train",
                             batch_size=self.dataset_config["batch_size"],
                             workers=self.dataset_config["workers"],
                             input_type=self.model_config["input_type"],
                             shuffle=True,
                             pin_mem=True,
                             clips_per_video=self.dataset_config["clips_per_video"],
                             augmentation_config=self.dataset_config["augmentation_config"])
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation."""
        val_loader = build_dataloader(sample_dict=self.val_dict,
                                      model_config=self.model_config,
                                      output_shape=self.data_shape,
                                      label_map=self.dataset_config["label_map"],
                                      dataset_mode="val",
                                      batch_size=self.dataset_config["batch_size"],
                                      workers=self.dataset_config["workers"],
                                      input_type=self.model_config["input_type"],
                                      clips_per_video=1,
                                      augmentation_config=self.dataset_config["augmentation_config"]
                                      )
        return val_loader

    def configure_optimizers(self):
        """Configure optimizers for training"""
        self.train_config = self.experiment_spec["train"]
        optim_dict = {}
        optim = torch.optim.SGD(params=self.parameters(),
                                lr=self.train_config['optim']['lr'],
                                momentum=self.train_config['optim']['momentum'],
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
            raise ValueError("Only [AutoReduce, MultiStep] scheduler is supported")

        optim_dict["lr_scheduler"] = lr_scheduler
        optim_dict['monitor'] = self.train_config['optim']['lr_monitor']

        return optim_dict

    def training_step(self, batch, batch_idx):
        """Training step."""
        data, label = batch
        output = self.model(data)
        loss = F.cross_entropy(output, label)
        self.train_accuracy.update(output, label)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc_1", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        _, data, label = batch
        output = self.model(data)
        loss = F.cross_entropy(output, label)
        self.val_accuracy.update(output, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc_1", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        """Log Validation metrics to status.json"""
        average_val_loss = 0.0
        for out in validation_step_outputs:
            average_val_loss += out.item()
        average_val_loss /= len(validation_step_outputs)

        self.status_logging_dict["val_loss"] = average_val_loss
        self.status_logging_dict["val_acc"] = self.val_accuracy.compute().item()

    def forward(self, x):
        """Forward of the action recognition model."""
        output = self.model(x)
        return output

    # @rank_zero_only
    # def training_epoch_end(self, outputs: List[Any]) -> None:
    #     pass

    # @rank_zero_only
    # def validation_epoch_end(self, outputs: List[Any]) -> None:
    #     pass

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
