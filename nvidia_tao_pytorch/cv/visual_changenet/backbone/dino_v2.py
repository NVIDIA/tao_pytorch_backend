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

""" DINOv2 ViT Model Module """

from functools import partial

from nvidia_tao_pytorch.cv.backbone_v2.dino_v2 import vit_large_patch14_dinov2_swiglu


vit_model_dict = {
    'vit_large_nvdinov2': partial(vit_large_patch14_dinov2_swiglu, num_classes=0)
}
