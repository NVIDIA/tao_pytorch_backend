# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Backbone module.

TODO(yuw, hongyu): Replace `nvidia_tao_pytorch.cv.backbone` with
`nvidia_tao_pytorch.cv.backbone_v2` once all backbones are unified.
"""
from nvidia_tao_pytorch.cv.backbone_v2.registry import BACKBONE_REGISTRY

from nvidia_tao_pytorch.cv.backbone_v2.fastervit import (
    faster_vit_0_224,
    faster_vit_1_224,
    faster_vit_2_224,
    faster_vit_3_224,
    faster_vit_4_224,
    faster_vit_5_224,
    faster_vit_6_224,
    faster_vit_4_21k_224,
    faster_vit_4_21k_384,
    faster_vit_4_21k_512,
    faster_vit_4_21k_768,
)
from nvidia_tao_pytorch.cv.backbone_v2.fan import (
    fan_tiny_12_p16_224,
    fan_small_12_p16_224_se_attn,
    fan_small_12_p16_224,
    fan_base_18_p16_224,
    fan_large_24_p16_224,
    fan_tiny_8_p4_hybrid,
    fan_small_12_p4_hybrid,
    fan_base_16_p4_hybrid,
    fan_large_16_p4_hybrid,
    fan_xlarge_16_p4_hybrid,
    fan_swin_tiny_patch4_window7_224,
    fan_swin_small_patch4_window7_224,
    fan_swin_base_patch4_window7_224,
    fan_swin_large_patch4_window7_224,
)
from nvidia_tao_pytorch.cv.backbone_v2.dino_v2 import (
    vit_large_patch14_dinov2_swiglu,
    vit_large_patch14_dinov2_swiglu_legacy,
    vit_giant_patch14_reg4_dinov2_swiglu,
)
from nvidia_tao_pytorch.cv.backbone_v2.efficientvit import (
    efficientvit_b0,
    efficientvit_b1,
    efficientvit_b2,
    efficientvit_b3,
    efficientvit_l0,
    efficientvit_l1,
    efficientvit_l2,
    efficientvit_l3,
)
from nvidia_tao_pytorch.cv.backbone_v2.vit import (
    vit_base_patch16,
    vit_large_patch16,
    vit_huge_patch14,
)
from nvidia_tao_pytorch.cv.backbone_v2.convnext import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    convnext_xlarge,
)
from nvidia_tao_pytorch.cv.backbone_v2.convnext_v2 import (
    convnextv2_atto,
    convnextv2_femto,
    convnextv2_pico,
    convnextv2_nano,
    convnextv2_tiny,
    convnextv2_base,
    convnextv2_large,
    convnextv2_huge,
)
from nvidia_tao_pytorch.cv.backbone_v2.hiera import (
    hiera_tiny_224,
    hiera_small_224,
    hiera_base_224,
    hiera_base_plus_224,
    hiera_large_224,
    hiera_huge_224,
)
from nvidia_tao_pytorch.cv.backbone_v2.resnet import (
    resnet_18,
    resnet_34,
    resnet_50,
    resnet_101,
    resnet_152,
    resnet_18d,
    resnet_34d,
    resnet_50d,
    resnet_101d,
    resnet_152d,
)
from nvidia_tao_pytorch.cv.backbone_v2.swin import (
    swin_tiny_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_base_patch4_window7_224,
    swin_large_patch4_window7_224,
    swin_base_patch4_window12_384,
    swin_large_patch4_window12_384,
)
from nvidia_tao_pytorch.cv.backbone_v2.gcvit import (
    gc_vit_xxtiny,
    gc_vit_xtiny,
    gc_vit_tiny,
    gc_vit_small,
    gc_vit_base,
    gc_vit_large,
    gc_vit_base_384,
    gc_vit_large_384,
)
from nvidia_tao_pytorch.cv.backbone_v2.edgenext import (
    edgenext_xx_small,
    edgenext_x_small,
    edgenext_small,
    edgenext_base,
    edgenext_xx_small_bn_hs,
    edgenext_x_small_bn_hs,
    edgenext_small_bn_hs,
)
from nvidia_tao_pytorch.cv.backbone_v2.radio import (
    c_radio_p1_vit_huge_patch16_mlpnorm,
    c_radio_p2_vit_huge_patch16_mlpnorm,
    c_radio_p3_vit_huge_patch16_mlpnorm,
    c_radio_v2_vit_base_patch16,
    c_radio_v2_vit_large_patch16,
    c_radio_v2_vit_huge_patch16,
    c_radio_v3_vit_large_patch16_reg4_dinov2,
)
from nvidia_tao_pytorch.cv.backbone_v2.open_clip import (
    vit_l_14_siglip_clipa_224,
    vit_l_14_siglip_clipa_336,
    vit_h_14_siglip_clipa_224,
)

__all__ = [
    "BACKBONE_REGISTRY",
    "faster_vit_0_224",
    "faster_vit_1_224",
    "faster_vit_2_224",
    "faster_vit_3_224",
    "faster_vit_4_224",
    "faster_vit_5_224",
    "faster_vit_6_224",
    "faster_vit_4_21k_224",
    "faster_vit_4_21k_384",
    "faster_vit_4_21k_512",
    "faster_vit_4_21k_768",
    "fan_tiny_12_p16_224",
    "fan_small_12_p16_224_se_attn",
    "fan_small_12_p16_224",
    "fan_base_18_p16_224",
    "fan_large_24_p16_224",
    "fan_tiny_8_p4_hybrid",
    "fan_small_12_p4_hybrid",
    "fan_base_16_p4_hybrid",
    "fan_large_16_p4_hybrid",
    "fan_xlarge_16_p4_hybrid",
    "fan_swin_tiny_patch4_window7_224",
    "fan_swin_small_patch4_window7_224",
    "fan_swin_base_patch4_window7_224",
    "fan_swin_large_patch4_window7_224",
    "vit_large_patch14_dinov2_swiglu",
    "vit_large_patch14_dinov2_swiglu_legacy",
    "vit_giant_patch14_reg4_dinov2_swiglu",
    "efficientvit_b0",
    "efficientvit_b1",
    "efficientvit_b2",
    "efficientvit_b3",
    "efficientvit_l0",
    "efficientvit_l1",
    "efficientvit_l2",
    "efficientvit_l3",
    "vit_base_patch16",
    "vit_large_patch16",
    "vit_huge_patch14",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "convnext_xlarge",
    "convnextv2_atto",
    "convnextv2_femto",
    "convnextv2_pico",
    "convnextv2_nano",
    "convnextv2_tiny",
    "convnextv2_base",
    "convnextv2_large",
    "convnextv2_huge",
    "hiera_tiny_224",
    "hiera_small_224",
    "hiera_base_224",
    "hiera_base_plus_224",
    "hiera_large_224",
    "hiera_huge_224",
    "resnet_18",
    "resnet_34",
    "resnet_50",
    "resnet_101",
    "resnet_152",
    "resnet_18d",
    "resnet_34d",
    "resnet_50d",
    "resnet_101d",
    "resnet_152d",
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
    "swin_large_patch4_window7_224",
    "swin_base_patch4_window12_384",
    "swin_large_patch4_window12_384",
    "gc_vit_xxtiny",
    "gc_vit_xtiny",
    "gc_vit_tiny",
    "gc_vit_small",
    "gc_vit_base",
    "gc_vit_large",
    "gc_vit_base_384",
    "gc_vit_large_384",
    "edgenext_xx_small",
    "edgenext_x_small",
    "edgenext_small",
    "edgenext_base",
    "edgenext_xx_small_bn_hs",
    "edgenext_x_small_bn_hs",
    "edgenext_small_bn_hs",
    "c_radio_p1_vit_huge_patch16_mlpnorm",
    "c_radio_p2_vit_huge_patch16_mlpnorm",
    "c_radio_p3_vit_huge_patch16_mlpnorm",
    "c_radio_v2_vit_base_patch16",
    "c_radio_v2_vit_large_patch16",
    "c_radio_v2_vit_huge_patch16",
    "c_radio_v3_vit_large_patch16_reg4_dinov2",
    "vit_l_14_siglip_clipa_224",
    "vit_l_14_siglip_clipa_336",
    "vit_h_14_siglip_clipa_224",
]
