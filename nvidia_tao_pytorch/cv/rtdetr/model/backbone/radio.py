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

"""RADIO backbone for RT-DETR."""

from argparse import Namespace
from typing import Dict, Tuple, Type

import torch

from nvidia_tao_pytorch.cv.backbone_v2.radio import (
    RADIO,
    c_radio_v2_vit_base_patch16,
    c_radio_v2_vit_large_patch16,
    c_radio_v2_vit_huge_patch16,
    c_radio_v3_vit_base_patch16_reg4_dinov2,
    c_radio_v3_vit_large_patch16_reg4_dinov2,
    c_radio_v3_vit_huge_patch16_reg4_dinov2
)
torch.serialization.add_safe_globals([Namespace])


radio_model_dict: Dict[str, Tuple[Type[RADIO], Tuple[int, int]]] = {
    # encoder_channel, decoder_channel
    # "e-radio_v2": (1536, 1536),
    # "radio_v2.5-b": (768, 2304),
    # "radio_v2.5-l": (1024, 3072),
    # "radio_v2.5-h": (1280, 3840),
    "radio_v2-b": [c_radio_v2_vit_base_patch16, (768, 2304)],
    "radio_v2-l": [c_radio_v2_vit_large_patch16, (1024, 3072)],
    "radio_v2-h": [c_radio_v2_vit_huge_patch16, (1280, 3840)],
    "radio_v3-b": [c_radio_v3_vit_base_patch16_reg4_dinov2, (768, 2304)],
    "radio_v3-l": [c_radio_v3_vit_large_patch16_reg4_dinov2, (1024, 3072)],
    "radio_v3-h": [c_radio_v3_vit_huge_patch16_reg4_dinov2, (1280, 3840)],
}
