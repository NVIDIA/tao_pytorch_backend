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

"""Grounding DINO model utils. """
from nvidia_tao_pytorch.core.utils.ptm_utils import StateDictAdapter

ptm_adapter = StateDictAdapter()
ptm_adapter.add("mae", "model.encoder.")
ptm_adapter.add("classification", "model.")
ptm_adapter.add("grounding_dino", "model.")


def grounding_dino_parser(original):
    """Parse public Grounding DINO checkpoints.

    Download checkpoints from https://github.com/IDEA-Research/GroundingDINO/releases.
    """
    final = {}
    for k, v in original.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
        if k.startswith('backbone.0'):
            k = f"model.model.{k}"
            k = k.replace("backbone.0", "backbone.0.body")
        elif k == "bert.embeddings.position_ids":
            continue
        elif "label_enc.weight" in k:
            continue
        else:
            k = f"model.model.{k}"
        final[k] = v
    return final
