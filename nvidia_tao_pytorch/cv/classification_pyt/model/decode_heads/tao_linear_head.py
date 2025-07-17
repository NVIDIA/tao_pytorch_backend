# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" FAN Linear Class Head """

import torch
import torch.nn as nn

from typing import Tuple


class TAOLinearClsHead(nn.Module):
    """Linear classifier head updated from MMPretrain to fix Feat return Bug.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 binary,
                 num_classes,
                 in_channels,
                 head_init_scale=None
                 ):
        """ Init Module """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.head_init_scale = head_init_scale
        self.binary = binary

        if self.num_classes < 0:
            raise ValueError(
                f'num_classes={num_classes} must be non-negative')

        if self.num_classes != 2 and self.binary:
            raise ValueError(
                f'Only support binary head when num_classes == 2, Got num_classes == {self.num_classes}'
            )

        if self.num_classes == 0:
            self.fc = nn.Identity()
        else:
            if self.binary:
                self.fc = nn.Linear(self.in_channels, 1)
            else:
                self.fc = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        # The final classification head.
        cls_score = self.fc(feats)

        return cls_score
