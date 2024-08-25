# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Common Lightning Module"""

from typing import Any, Dict, Sequence
import re

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.utilities import get_latest_checkpoint, patch_decrypt_checkpoint


class TAOLightningModule(pl.LightningModule):
    """Common PTL module"""

    def __init__(self, experiment_spec, **kwargs):
        """Init training"""
        super().__init__(**kwargs)
        self.experiment_spec = experiment_spec
        self.dataset_config = experiment_spec["dataset"]
        self.model_config = experiment_spec["model"]

        self.checkpoint_filename = None

    def configure_callbacks(self) -> Sequence[Callback] | pl.Callback:
        """Configures logging and checkpoint-saving callbacks"""
        # This is called when trainer.fit() is called

        results_dir = self.experiment_spec["results_dir"]
        num_epochs = self.experiment_spec["train"]["num_epochs"]
        checkpoint_interval = self.experiment_spec["train"]["checkpoint_interval"]

        status_logger_callback = TAOStatusLogger(
            results_dir,
            append=True,
            num_epochs=num_epochs
        )

        resume_ckpt = self.experiment_spec["train"]["resume_training_checkpoint_path"] or get_latest_checkpoint(results_dir)
        if resume_ckpt:
            resumed_epoch = re.search('epoch_(\\d+)', resume_ckpt)
            if resumed_epoch:
                resumed_epoch = int(resumed_epoch.group(1))
        else:
            resumed_epoch = 0
        status_logger_callback.epoch_counter = resumed_epoch + 1

        ModelCheckpoint.FILE_EXTENSION = ".pth"
        ModelCheckpoint.CHECKPOINT_EQUALS_CHAR = "_"

        if not self.checkpoint_filename:
            raise NotImplementedError("checkpoint_filename not set in __init__() of model")
        ModelCheckpoint.CHECKPOINT_NAME_LAST = f"{self.checkpoint_filename}_latest"

        checkpoint_callback = ModelCheckpoint(every_n_epochs=checkpoint_interval,
                                              dirpath=results_dir,
                                              save_on_train_epoch_end=True,
                                              monitor=None,
                                              save_top_k=-1,
                                              save_last='link',
                                              filename='model_{epoch:03d}',
                                              enable_version_counter=False)

        return [status_logger_callback, checkpoint_callback]

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
