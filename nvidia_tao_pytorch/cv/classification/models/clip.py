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

""" open_clip Model Module """

from torch import nn
from mmcls.models.builder import BACKBONES
from open_clip import create_model


@BACKBONES.register_module()
class open_clip(nn.Module):
    """
    open_clip model
    """

    def __init__(self, model_name="ViT-B-32", freeze=False, **kwargs):
        """
        Constructor for open_clip model
        """
        super().__init__()

        self.model = create_model(model_name, **kwargs)
        self.freeze = freeze

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward function and return the features
        """
        if self.freeze:
            self.eval()

        return self.model.encode_image(x, normalize=False)
