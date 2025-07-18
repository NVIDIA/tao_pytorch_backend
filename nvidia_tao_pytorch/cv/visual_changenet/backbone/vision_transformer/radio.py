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

""" RADIO ViT Model Module """

from nvidia_tao_pytorch.cv.classification_pyt.model.backbones.radio import (
    CRadioP1VitHugePatch16Mlpnorm,
    CRadioP2VitHugePatch16Mlpnorm,
    CRadioP3VitHugePatch16Mlpnorm,
    CRadioV2VitHugePatch16,
    CRadioV2VitLargePatch16,
    CRadioV2VitBasePatch16
)

radio_model_dict = {
    'c_radio_p1_vit_huge_patch16_224_mlpnorm': CRadioP1VitHugePatch16Mlpnorm,
    'c_radio_p2_vit_huge_patch16_224_mlpnorm': CRadioP2VitHugePatch16Mlpnorm,
    'c_radio_p3_vit_huge_patch16_224_mlpnorm': CRadioP3VitHugePatch16Mlpnorm,
    'c_radio_v2_vit_huge_patch16_224': CRadioV2VitHugePatch16,
    'c_radio_v2_vit_large_patch16_224': CRadioV2VitLargePatch16,
    'c_radio_v2_vit_base_patch16_224': CRadioV2VitBasePatch16
}
