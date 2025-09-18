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

"""RADIO ViT Model Module"""

from nvidia_tao_pytorch.cv.backbone_v2.radio import (
    c_radio_p1_vit_huge_patch16_mlpnorm,
    c_radio_p2_vit_huge_patch16_mlpnorm,
    c_radio_p3_vit_huge_patch16_mlpnorm,
    c_radio_v2_vit_base_patch16,
    c_radio_v2_vit_huge_patch16,
    c_radio_v2_vit_large_patch16,
)


radio_model_dict = {
    "c_radio_p1_vit_huge_patch16_224_mlpnorm": c_radio_p1_vit_huge_patch16_mlpnorm,
    "c_radio_p2_vit_huge_patch16_224_mlpnorm": c_radio_p2_vit_huge_patch16_mlpnorm,
    "c_radio_p3_vit_huge_patch16_224_mlpnorm": c_radio_p3_vit_huge_patch16_mlpnorm,
    "c_radio_v2_vit_huge_patch16_224": c_radio_v2_vit_huge_patch16,
    "c_radio_v2_vit_large_patch16_224": c_radio_v2_vit_large_patch16,
    "c_radio_v2_vit_base_patch16_224": c_radio_v2_vit_base_patch16,
}
