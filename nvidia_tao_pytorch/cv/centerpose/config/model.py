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

from nvidia_tao_pytorch.config.types import (
    BOOL_FIELD,
    INT_FIELD,
    STR_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class BackboneConfig:
    """CenterPose backbone model config."""

    model_type: str = STR_FIELD(
        value="fan_small",
        description="Model type.",
        display_name="Model Type",
        popular="fan_small"
    )
    pretrained_backbone_path: str = STR_FIELD(
        value="",
        description="Path to the pretrained backbone model.",
        display_name="Pretrained Backbone Model"
    )


@dataclass
class CenterPoseModelConfig:
    """CenterPose model config."""

    down_ratio: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=4,
        valid_max=4,
        description="Down ratio.",
        display_name="Down Ratio",
        popular="4"
    )
    final_kernel: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max=1,
        description="Final kernel size.",
        display_name="Final Kernel Size",
        popular="1"
    )
    last_level: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=5,
        valid_max=5,
        description="Last level.",
        display_name="Last Level",
        popular="5"
    )
    head_conv: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=256,
        valid_max=256,
        description="Head convolution.",
        display_name="Head Convolution",
        popular="256"
    )
    out_channel: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max=0,
        description="Output channel.",
        display_name="Output Channel"
    )
    use_convGRU: bool = BOOL_FIELD(
        value=True,
        description="Use convolutional GRU.",
        display_name="Convolutional GRU"
    )
    use_pretrained: bool = BOOL_FIELD(
        value=False,
        description="Use pretrained model.",
        display_name="Pretrained Model"
    )
    backbone: BackboneConfig = DATACLASS_FIELD(
        BackboneConfig(),
        description="Backbone model config.",
        display_name="Backbone Model"
    )
