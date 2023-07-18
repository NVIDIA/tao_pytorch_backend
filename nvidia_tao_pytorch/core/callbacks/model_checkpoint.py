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

"""File containing Overloaded version of PTL ModelCheckpoint."""

import os
from copy import deepcopy

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from nvidia_tao_pytorch.core.checkpoint_encryption import decrypt_checkpoint
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook


class TLTModelCheckpoint(ModelCheckpoint):
    """Light wrapper around Lightning's ModelCheckpoint to force a saved checkpoint on train_end.

    Args:
        prefix (str): Prefix string to be added to the model. Default=model
        save_best_model (bool): Flag to save the best model. Default=True
        postfix (str): File extension to save the model.
    """

    def __init__(self, prefix="model", save_best_model=True, postfix=".etlt", **kwargs):
        """Constructor function for the class."""
        # Call the parent class constructor with the remaining kwargs.
        super().__init__(**kwargs)

        # Parse and store "extended" parameters: save_best model and postfix.
        self.save_best_model = save_best_model
        self.postfix = postfix
        self.previous_best_path = ""
        # self.filename = prefix
        self.prefix = prefix

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Functions called by PTL when saving checkpoint. Used to save encypted models with EFF

        This function is only run for the rank 0 process in a multiGPU training.

        Args:
            trainer (pytorch_lightning.Trainer): PTL trainer calling the checkpoint callback.
            pl_module (pytorch_lightning.LightningModule): Lightning module implementing the model.
            checkpoint (dict): Pytorch lightning checkpoint dictionary.

        Return:
            output (LightningModule.state_dict): Checkpoint containing encrypted state dict.
        """
        output = super().state_dict(trainer, pl_module, checkpoint)

        # Load the best model and then re-save it
        if self.save_best_model:
            if not os.path.exists(self.best_model_path):
                return output

            if self.best_model_path == self.previous_best_path:
                return output

            self.previous_model_path = self.best_model_path
            old_state_dict = deepcopy(pl_module.state_dict())

            checkpoint = torch.load(self.best_model_path, map_location='cpu')
            checkpoint = decrypt_checkpoint(checkpoint, TLTPyTorchCookbook.get_passphrase())
            # trainer._checkpoint_connector.restore_model_state(pl_module, checkpoint)

            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            # get a new instanace of the model
            # TLTPyTorchCookbook().restore_from_ckpt()
            pl_module.load_state_dict(checkpoint)
            TLTPyTorchCookbook().save_checkpoint_to(pl_module.netG.module.state_dict(), save_path=os.path.join(self.dirpath, self.prefix + "_best" + self.postfix))
            # pl_module.save_to(save_path=os.path.join(self.dirpath, self.prefix + self.postfix))
            pl_module.load_state_dict(old_state_dict, strict=True)
        else:
            TLTPyTorchCookbook().save_checkpoint_to(pl_module.netG.module.state_dict(), save_path=os.path.join(self.dirpath, self.prefix + self.postfix))

            # pl_module.save_to(save_path=os.path.join(self.dirpath, self.prefix + self.postfix))

        return output

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """Overriden PTL function to force save of EFF encrypted model.

        Args:
            trainer (pytorch_lightning.Trainer): PTL trainer calling the checkpoint callback.
            pl_module (pytorch_lightning.LightningModule): Lightning module implementing the model.
        """
        if trainer.fast_dev_run:
            return None
        # Load the best model and then re-save it
        if self.save_best_model:
            checkpoint = torch.load(self.best_model_path, map_location='cpu')
            checkpoint = decrypt_checkpoint(checkpoint, TLTPyTorchCookbook.get_passphrase())
            # trainer._checkpoint_connector.restore(self.best_model_path, on_gpu=trainer.on_gpu)
        TLTPyTorchCookbook().save_checkpoint_to(pl_module.netG.module.state_dict(), save_path=os.path.join(self.dirpath, self.prefix + "_last" + self.postfix))
        return None
