# The code defines a backbone model for distilling TAO Toolkit models using the timm library.
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

"""Core model components for TAO Toolkit models."""
from typing import List, Optional

import torch
import torch.nn as nn

import timm


class Backbone(nn.Module):
    """Base backbone module"""

    def out_strides(self) -> List[int]:
        """Returns the output strides of the backbone"""
        raise NotImplementedError

    def out_channels(self) -> List[int]:
        """Returns the output channels of the backbone"""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the backbone"""
        raise NotImplementedError


class TimmBackbone(Backbone):
    """TimmBackbone class to use timm for creating backbone

    Uses model_name to create backbone and downloads pretrained weights if specified
    """

    def __init__(self, model_name: str,
                 pretrained: bool = True,
                 out_indices: Optional[List[int]] = None,
                 pretrained_path: Optional[str] = None
                 ):
        """Initializes the TimmBackbone"""
        super().__init__()

        if pretrained_path:
            self.model = timm.create_model(
                model_name=model_name,
                pretrained=False,
                features_only=True,
                out_indices=out_indices,
                pretrained_cfg_overlay=dict(file=pretrained_path)
            )
        else:
            self.model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices
            )

    def out_strides(self) -> List[int]:
        """Returns the output strides of the backbone"""
        return self.model.feature_info.reduction()

    def out_channels(self) -> List[int]:
        """Returns the output channels of the backbone"""
        return self.model.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the backbone"""
        return self.model(x)
