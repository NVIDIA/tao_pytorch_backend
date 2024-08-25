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

"""Distiller core for TAO Toolkit models."""
from abc import abstractmethod
from typing import Any, Dict

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule


class Distiller(TAOLightningModule):
    """PTL Module for Distillation."""

    def __init__(self, experiment_spec, export=False):
        """Initializes the distiller from given experiment_spec."""
        super().__init__(experiment_spec)

        self.distillation_config = experiment_spec.distill
        self.eval_class_ids = self.dataset_config["eval_class_ids"]
        self.dataset_type = self.dataset_config["dataset_type"]
        if self.dataset_type not in ("serialized", "default"):
            raise ValueError(f"{self.dataset_type} is not supported. Only serialized and default are supported.")

        self.status_logging_dict = {}

        self._build_model(export)
        self._build_criterion()

    @abstractmethod
    def _build_model(self, export):
        """Internal function to build the model."""
        # Should instantiate student and teacher model from experiment_spec
        raise NotImplementedError('Subclasses must implement _build_model')

    @abstractmethod
    def _build_criterion(self):
        """Internal function to build the loss function."""
        raise NotImplementedError('Subclasses must implement _build_criterion')

    @abstractmethod
    def configure_optimizers(self):
        """Internal function to build the optimizer."""
        self.train_config = self.experiment_spec.train

    @abstractmethod
    def on_train_epoch_start(self):
        """Train epoch start. Declaring output list."""
        # raise NotImplementedError('Subclasses must implement on_train_epoch_start')
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Internal function to define the training step."""
        raise NotImplementedError('Subclasses must implement training_step')

    @abstractmethod
    def on_train_epoch_end(self, training_step_outputs):
        """Internal function to define the training epoch end."""
        raise NotImplementedError('Subclasses must implement training_epoch_end')

    @abstractmethod
    def on_validation_epoch_start(self) -> None:
        """Internal function to define the validation epoch start."""
        raise NotImplementedError('Subclasses must implement on_validation_epoch_start')

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """Internal function to define the validation step."""
        raise NotImplementedError('Subclasses must implement validation_step')

    @abstractmethod
    def on_validation_epoch_end(self):
        """Internal function to define the validation epoch end."""
        raise NotImplementedError('Subclasses must implement validation_epoch_end')

    @abstractmethod
    def on_test_epoch_start(self) -> None:
        """Internal function to define the test epoch start."""
        # raise NotImplementedError('Subclasses must implement on_test_epoch_start')
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx):
        """Internal function to define the test step."""
        # raise NotImplementedError('Subclasses must implement test_step')
        pass

    @abstractmethod
    def on_test_epoch_end(self):
        """Internal function to define the test epoch end."""
        # raise NotImplementedError('Subclasses must implement test_epoch_end')
        pass

    @abstractmethod
    def predict_step(self, batch, batch_idx):
        """Internal function to define the predict step."""
        # raise NotImplementedError('Subclasses must implement predict_step')
        pass

    @abstractmethod
    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Internal function to define the predict batch end."""
        # raise NotImplementedError('Subclasses must implement on_predict_batch_end')
        pass

    @abstractmethod
    def forward(self, x):
        """Internal function to define the forward pass."""
        raise NotImplementedError('Subclasses must implement forward')

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Encrpyt the checkpoint. The encryption is done in TLTCheckpointConnector."""
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Decrpyt the checkpoint."""
        pass
