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

from detectron2.config import instantiate


def instantiate_odise(cfg):
    """Instantiate model from config file."""
    backbone = instantiate(cfg.backbone)

    cfg.sem_seg_head.pixel_decoder.input_shape = {
        k: v for k, v in backbone.output_shape().items() if k in ["res2", "res3", "res4", "res5"]
    }
    cfg.sem_seg_head.input_shape = {
        k: v for k, v in backbone.output_shape().items() if k in ["res2", "res3", "res4", "res5"]
    }
    cfg.backbone = backbone
    model = instantiate(cfg)

    return model
