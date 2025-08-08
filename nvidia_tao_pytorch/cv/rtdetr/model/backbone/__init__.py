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

"""Backbones for RT-DETR."""

from nvidia_tao_pytorch.cv.rtdetr.model.backbone.registry import RTDETR_BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.resnet import (
    resnet_18,
    resnet_34,
    resnet_50,
    resnet_101,
)
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.convnext import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    convnext_xlarge
)
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.convnext_v2 import (
    convnextv2_nano,
    convnextv2_tiny,
    convnextv2_base,
    convnextv2_large,
    convnextv2_huge,
)
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.efficientvit import (
    efficientvit_b0,
    efficientvit_b1,
    efficientvit_b2,
    efficientvit_b3,
    efficientvit_l0,
    efficientvit_l1,
    efficientvit_l2,
    efficientvit_l3,
)
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.fan import (
    fan_tiny_8_p4_hybrid,
    fan_small_12_p4_hybrid,
    fan_base_12_p4_hybrid,
    fan_large_12_p4_hybrid,
)
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.edgenext import (
    edgenext_x_small,
    edgenext_small,
    edgenext_base,
)

__all__ = [
    "RTDETR_BACKBONE_REGISTRY",
    "resnet_18",
    "resnet_34",
    "resnet_50",
    "resnet_101",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "convnext_xlarge",
    "convnextv2_nano",
    "convnextv2_tiny",
    "convnextv2_base",
    "convnextv2_large",
    "convnextv2_huge",
    "efficientvit_b0",
    "efficientvit_b1",
    "efficientvit_b2",
    "efficientvit_b3",
    "efficientvit_l0",
    "efficientvit_l1",
    "efficientvit_l2",
    "efficientvit_l3",
    "fan_tiny_8_p4_hybrid",
    "fan_small_12_p4_hybrid",
    "fan_base_12_p4_hybrid",
    "fan_large_12_p4_hybrid",
    "edgenext_x_small",
    "edgenext_small",
    "edgenext_base",
]
