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

"""RADIO model base class
"""

from nvidia_tao_pytorch.cv.backbone.radio.utils import get_prefix_state_dict
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank


import logging
import torch
from torch import nn
from abc import abstractmethod

logger = logging.getLogger(__name__)


class RADIOBase(nn.Module):
    """RADIO base class
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def num_summary_tokens(self) -> int:
        """number of summary tokens (include register tokens and class tokens)
        """
        raise NotImplementedError('Subclasses must have property of num_summary_tokens')

    @property
    @abstractmethod
    def patch_size(self) -> int:
        """ViT patch size
        """
        raise NotImplementedError('Subclasses must have property of patch_size')

    @property
    def window_size(self) -> int:
        """Windown size for windowed attention
        """
        raise NotImplementedError('Subclasses must have property of window_size')

    @abstractmethod
    def _build_model(self, model_name: str):
        """Internal function to build the model

        Args:
            model_name (str): model to be built

        Raises:
            NotImplementedError
        """
        raise NotImplementedError('Subclasses must implement _build_model')

    def load_state_dict(self, checkpoint: str):
        """Customize weight loading for RADIO

        Args:
            checkpoint (str): checkpoint path
        """
        chk = torch.load(checkpoint, map_location="cpu")
        key_warn = self.model.load_state_dict(get_prefix_state_dict(chk, "base_model."), strict=False)

        if get_global_rank() == 0:
            if key_warn.missing_keys:
                logger.info(f"Missing keys in state dict: {key_warn.missing_keys}")

            if key_warn.unexpected_keys:
                logger.info(f"Unexpected keys in state dict: {key_warn.unexpected_keys}")
