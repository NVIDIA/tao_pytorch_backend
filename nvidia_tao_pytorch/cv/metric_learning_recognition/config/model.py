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

"""Configuration hyperparameter schema for the model."""

from dataclasses import dataclass
from typing import Optional

from nvidia_tao_pytorch.config.types import (
    INT_FIELD,
    STR_FIELD,
)


@dataclass
class MLModelConfig:
    """Metric Learning Recognition model configuration for training, testing & validation."""

    backbone: str = STR_FIELD(
        value="resnet_50",
        default_value="resnet_50",
        display_name="backbone",
        description="The backbone name of the model",
        valid_options=",".join(["resnet_50", "resnet_101", "fan_small", "fan_base", "fan_large",
                               "fan_tiny", "nvdinov2_vit_large_legacy"])
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="pretrained model path",
        description="[Optional] Path to the pretrained model. The weights are only loaded to the whole model.",
    )
    pretrained_trunk_path: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="pretrained_trunk_path",
        description="[Optional] Path to the pretrained trunk. The weights are only loaded to the trunk part.",
    )
    pretrained_embedder_path: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="pretrained_embedder_path",
        description="[Optional] Path to the pretrained embedder. The weights are only loaded to the embedder part.",
    )
    input_width: int = INT_FIELD(
        value=224,
        default_value=224,
        description="The input width of the images.",
        display_name="input_width",
    )
    input_height: int = INT_FIELD(
        value=224,
        default_value=224,
        description="The input height of the images.",
        display_name="input_height",
    )
    input_channels: int = INT_FIELD(
        value=3,
        default_value=3,
        description="The number of input channels.",
        display_name="input_channels",
    )
    feat_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="The output size of the feature embeddings.",
        display_name="feature dimension",
    )
