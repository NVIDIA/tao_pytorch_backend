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

""" Model Parameters Mapping Module """

map_params = {"head": {"in_channels": {
    "fan_tiny_8_p4_hybrid": 192,  # FAN
    "fan_small_12_p4_hybrid": 384,
    "fan_base_16_p4_hybrid": 448,
    "fan_large_16_p4_hybrid": 480,
    "fan_Xlarge_16_p4_hybrid": 768,
    "fan_base_18_p16_224": 448,
    "fan_tiny_12_p16_224": 192,
    "fan_small_12_p16_224_se_attn": 384,
    "fan_small_12_p16_224": 384,
    "fan_large_24_p16_224": 480,
    "gc_vit_xxtiny": 512,  # GCViT
    "gc_vit_xtiny": 512,
    "gc_vit_tiny": 512,
    "gc_vit_small": 768,
    "gc_vit_base": 1024,
    "gc_vit_large": 1536,
    "gc_vit_large_384": 1536,
    "faster_vit_0_224": 512,  # FasterViT
    "faster_vit_1_224": 640,
    "faster_vit_2_224": 768,
    "faster_vit_3_224": 1024,
    "faster_vit_4_224": 1568,
    "faster_vit_5_224": 2560,
    "faster_vit_6_224": 2560,
    "faster_vit_4_21k_224": 1568,
    "faster_vit_4_21k_384": 1568,
    "faster_vit_4_21k_512": 1568,
    "faster_vit_4_21k_768": 1568,
}}}
