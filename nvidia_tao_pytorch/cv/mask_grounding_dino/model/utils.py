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
""" Misc functions. """
from nvidia_tao_pytorch.core.utils.ptm_utils import StateDictAdapter

ptm_adapter = StateDictAdapter()
ptm_adapter.add("mae", "model.encoder.")
ptm_adapter.add("classification", "model.")
ptm_adapter.add("grounding_dino", "model.")


def mask_gdino_parser(original):
    """Parse public Mask Grounding DINO checkpoints."""
    state_dict = {}
    for key, value in list(original.items()):
        if "module" in key:
            new_key = ".".join(key.split(".")[1:])
            state_dict[new_key] = value
        elif key.startswith("backbone."):
            # MMLab compatible weight loading
            new_key = key[9:]
            state_dict[new_key] = value
        elif key.startswith("ema_"):
            # Do not include ema params from MMLab
            continue
        else:
            state_dict[key] = value
    return state_dict
