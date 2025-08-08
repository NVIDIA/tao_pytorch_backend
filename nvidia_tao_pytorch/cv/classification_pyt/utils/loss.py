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

"""Loss Functions for Classification"""

import torch.nn as nn
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.dataset import NOCLASS_IDX


class Cross_Entropy(nn.Module):
    """
    Cross Entropy Loss with label smoothing

    Args:
        weight (Tensor): A manual rescaling weight given to each class.
        label_smoothing (float): The label smoothing value.
        soft (bool): If True, allow soft label from a teacher model.
    """

    def __init__(self, weight=None, label_smoothing=0.1, soft=False):
        super(Cross_Entropy, self).__init__()
        self.soft = soft
        if soft:
            self.loss = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            self.loss = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                reduction="mean",
                ignore_index=NOCLASS_IDX,
            )

    def forward(self, pred, target):
        """
        Forward pass for Cross_Entropy
        """
        return self.loss(pred, target)
