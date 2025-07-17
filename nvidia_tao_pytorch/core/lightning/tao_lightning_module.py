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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.callbacks.model_checkpoint import TAOExceptionCheckpoint
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.utilities import patch_decrypt_checkpoint


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
        checkpoint_interval = self.experiment_spec["train"]["checkpoint_interval"]

        status_logger_callback = TAOStatusLogger(
            results_dir,
            append=True
        )

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
                                              filename='model_{epoch:03d}_{step:05d}',
                                              enable_version_counter=False)

        # For now, we use our custom one since Lightning's callback for this is minimal
        TAOExceptionCheckpoint.FILE_EXTENSION = ModelCheckpoint.FILE_EXTENSION
        TAOExceptionCheckpoint.CHECKPOINT_NAME_LAST = ModelCheckpoint.CHECKPOINT_NAME_LAST
        exception_checkpoint_callback = TAOExceptionCheckpoint(dirpath=results_dir)

        return [status_logger_callback, checkpoint_callback, exception_checkpoint_callback]

    # These are necessary because we sometimes have drop_last=True for the dataloaders.
    # When the dataset is smaller than the batch size, this leads to Lightning not
    # doing the task since it can't fill up a batch. However, it reports completion,
    # not failure. So, we do a manaul check and throw an error.
    def _dataloader_batch_check(self, dataloader, task):
        batch_size = dataloader.batch_size
        # Using a BatchSampler
        if not batch_size:
            assert hasattr(dataloader, "batch_sampler"), "Loader should have batch sampler initiated if batch size isn't defined."
            batch_size = dataloader.batch_sampler.batch_size
        dataset_len = len(dataloader.dataset)
        total_batch_size = batch_size * self.trainer.num_devices

        if dataset_len < total_batch_size:
            raise ValueError(f"Dataset size ({dataset_len}) is smaller than the total batch size "
                             f"({total_batch_size}). Not enough data for {task}.")

    def on_fit_start(self):
        """Before training begins."""
        self._dataloader_batch_check(self.trainer.datamodule.train_dataloader(), "train")

    def on_validation_start(self):
        """Before validation begins."""
        self._dataloader_batch_check(self.trainer.datamodule.val_dataloader(), "validation")

    def on_test_start(self):
        """Before testing begins."""
        self._dataloader_batch_check(self.trainer.datamodule.test_dataloader(), "evaluation")

    def on_predict_start(self):
        """Before inference begins."""
        self._dataloader_batch_check(self.trainer.datamodule.predict_dataloader(), "inference")

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
