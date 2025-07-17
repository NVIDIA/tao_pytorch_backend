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

radio_model_cfg = {
    "c_radio_p1_vit_huge_patch16_224_mlpnorm": {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 3,
        "cpe_max_size": 2048,
        "register_multiple": 16
    },
    "c_radio_p2_vit_huge_patch16_224_mlpnorm": {
        "summary_idxs": [0, 1, 2, 3],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 16
    },
    "c_radio_p3_vit_huge_patch16_224_mlpnorm": {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 16
    },
    "c_radio_v2_vit_base_patch16_224": {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 8
    },
    "c_radio_v2_vit_large_patch16_224": {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 8
    },
    "c_radio_v2_vit_huge_patch16_224": {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 8
    }
}
