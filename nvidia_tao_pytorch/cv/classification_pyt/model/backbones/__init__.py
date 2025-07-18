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

"""Backbone Init Module."""

from nvidia_tao_pytorch.cv.classification_pyt.model.backbones.fan import FanTiny12P16224, FanSmall12P16224SeAttn, FanSmall12P16224, FanBase18P16224, FanLarge24P16224, FanTiny8P4Hybrid, FanSmall12P4Hybrid, FanBase16P4Hybrid, FanLarge16P4Hybrid, FanXlarge16P4Hybrid
from nvidia_tao_pytorch.cv.classification_pyt.model.backbones.dinov2_vit import VitLargePatch14Dinov2Swiglu, VitGiantPatch14Reg4Dinov2Swiglu
from nvidia_tao_pytorch.cv.classification_pyt.model.backbones.radio import CRadioP1VitHugePatch16Mlpnorm, CRadioP2VitHugePatch16Mlpnorm, CRadioP3VitHugePatch16Mlpnorm, CRadioV2VitHugePatch16, CRadioV2VitLargePatch16, CRadioV2VitBasePatch16
from nvidia_tao_pytorch.cv.classification_pyt.model.backbones.faster_vit import FasterVit0224, FasterVit1224, FasterVit2224, FasterVit3224, FasterVit4224, FasterVit5224, FasterVit6224, FasterVit421k224, FasterVit421k384, FasterVit421k512, FasterVit421k768
from nvidia_tao_pytorch.cv.classification_pyt.model.backbones.gc_vit import GcVitXxtiny, GcVitXtiny, GcVitTiny, GcVitSmall, GcVitBase, GcVitLarge, GcVitLarge384
from nvidia_tao_pytorch.cv.classification_pyt.model.backbones.clip import OpenClip
from nvidia_tao_pytorch.ssl.mae.model.convnextv2 import convnextv2_group
convnextv2_model_dict = {i.__name__: i for i in convnextv2_group}


fan_model_dict = {
    'fan_tiny_12_p16_224': FanTiny12P16224,
    'fan_small_12_p16_224_se_attn': FanSmall12P16224SeAttn,
    'fan_small_12_p16_224': FanSmall12P16224,
    'fan_base_18_p16_224': FanBase18P16224,
    'fan_large_24_p16_224': FanLarge24P16224,
    'fan_tiny_8_p4_hybrid': FanTiny8P4Hybrid,
    'fan_small_12_p4_hybrid': FanSmall12P4Hybrid,
    'fan_base_16_p4_hybrid': FanBase16P4Hybrid,
    'fan_large_16_p4_hybrid': FanLarge16P4Hybrid,
    'fan_Xlarge_16_p4_hybrid': FanXlarge16P4Hybrid
}

nvdino_model_dict = {
    'vit_large_patch14_dinov2_swiglu': VitLargePatch14Dinov2Swiglu,
    'vit_giant_patch14_reg4_dinov2_swiglu': VitGiantPatch14Reg4Dinov2Swiglu
}

cradio_model_dict = {
    'c_radio_p1_vit_huge_patch16_mlpnorm': CRadioP1VitHugePatch16Mlpnorm,
    'c_radio_p2_vit_huge_patch16_mlpnorm': CRadioP2VitHugePatch16Mlpnorm,
    'c_radio_p3_vit_huge_patch16_mlpnorm': CRadioP3VitHugePatch16Mlpnorm,
    'c_radio_v2_vit_base_patch16': CRadioV2VitBasePatch16,
    'c_radio_v2_vit_large_patch16': CRadioV2VitLargePatch16,
    'c_radio_v2_vit_huge_patch16': CRadioV2VitHugePatch16
}

faster_vit_model_dict = {
    'faster_vit_0_224': FasterVit0224,
    'faster_vit_1_224': FasterVit1224,
    'faster_vit_2_224': FasterVit2224,
    'faster_vit_3_224': FasterVit3224,
    'faster_vit_4_224': FasterVit4224,
    'faster_vit_5_224': FasterVit5224,
    'faster_vit_6_224': FasterVit6224,
    'faster_vit_4_21k_224': FasterVit421k224,
    'faster_vit_4_21k_384': FasterVit421k384,
    'faster_vit_4_21k_512': FasterVit421k512,
    'faster_vit_4_21k_768': FasterVit421k768,
}

gc_vit_model_dict = {
    'gc_vit_xxtiny': GcVitXxtiny,
    'gc_vit_xtiny': GcVitXtiny,
    'gc_vit_tiny': GcVitTiny,
    'gc_vit_small': GcVitSmall,
    'gc_vit_base': GcVitBase,
    'gc_vit_large': GcVitLarge,
    'gc_vit_large_384': GcVitLarge384
}

clip_model_dict = {
    'open_clip': OpenClip
}

# "fan_tiny_8_p4_hybrid": 192,  # FAN
# "fan_small_12_p4_hybrid": 384,
# "fan_base_16_p4_hybrid": 448,
# "fan_large_16_p4_hybrid": 480,
# "fan_Xlarge_16_p4_hybrid": 768,
# "fan_base_18_p16_224": 448,
# "fan_tiny_12_p16_224": 192,
# "fan_small_12_p16_224_se_attn": 384,
# "fan_small_12_p16_224": 384,
# "fan_large_24_p16_224": 480,
# "gc_vit_xxtiny": 512,  # GCViT
# "gc_vit_xtiny": 512,
# "gc_vit_tiny": 512,
# "gc_vit_small": 768,
# "gc_vit_base": 1024,
# "gc_vit_large": 1536,
# "gc_vit_large_384": 1536,
# "faster_vit_0_224": 512,  # FasterViT
# "faster_vit_1_224": 640,
# "faster_vit_2_224": 768,
# "faster_vit_3_224": 1024,
# "faster_vit_4_224": 1568,
# "faster_vit_5_224": 2560,
# "faster_vit_6_224": 2560,
# "faster_vit_4_21k_224": 1568,
# "faster_vit_4_21k_384": 1568,
# "faster_vit_4_21k_512": 1568,
# "faster_vit_4_21k_768": 1568,
# "vit_large_patch14_dinov2_swiglu": 1024,
# "vit_giant_patch14_reg4_dinov2_swiglu": 1536,
# "c_radio_p1_vit_huge_patch16_224_mlpnorm": 3840,
# "c_radio_p2_vit_huge_patch16_224_mlpnorm": 5120,
# "c_radio_p3_vit_huge_patch16_224_mlpnorm": 3840
# "ViT-H-14-SigLIP-CLIPA-224": 1024,
# "ViT-L-14-SigLIP-CLIPA-336": 768,
# "ViT-L-14-SigLIP-CLIPA-224": 768,

# not yet
# "ViT-L-14": 768,
# "ViT-B-16": 512,
# "ViT-L-14-336": 768,
# "ViT-g-14": 1024,
# "ViT-H-14": 1024,
# "EVA02-E-14-plus": 1024,
# "EVA02-E-14": 1024,
# "EVA02-L-14-336": 768,
# "EVA02-L-14": 768,
# "ViT-B-32": 512,
