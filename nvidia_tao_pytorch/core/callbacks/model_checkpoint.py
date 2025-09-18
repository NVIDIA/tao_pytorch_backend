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

"""File containing Overloaded version of PTL OnExceptionCheckpoint."""

import os
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, OnExceptionCheckpoint


class TAOExceptionCheckpoint(OnExceptionCheckpoint):
    """A custom checkpointing callback for when training is interrupted.

    Extending Lightning's OnExceptionCheckpoint since it (in v.2.3.0) only supports saving
    to a provided path. We want to extend its capabilities to also symlink the *_latest.pth
    file to this dumped checkpoint.
    """

    CHECKPOINT_NAME_LAST = ""

    def __init__(self, dirpath):
        """Callback init. Uses parent's default filename since we override below"""
        super().__init__(dirpath)

    def on_exception(self, trainer: "pl.Trainer", *_: Any, **__: Any) -> None:
        """Overriden function that saves and links the checkpoint"""
        self.filename = f"model_epoch_{trainer.current_epoch:03d}_step_{trainer.global_step:05d}"
        super().on_exception(trainer)
        ModelCheckpoint._link_checkpoint(trainer, self.ckpt_path, os.path.join(self.dirpath, self.CHECKPOINT_NAME_LAST + self.FILE_EXTENSION))
