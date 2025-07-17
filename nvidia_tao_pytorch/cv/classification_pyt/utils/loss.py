# Copyright (c) 2023 Chaminda Bandara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Original source taken from https://github.com/wgcban/ChangeFormer
#
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

"""Loss Functions for Classification"""

import torch.nn as nn
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.dataset import NOCLASS_IDX


class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with label smoothing

    Args:
        label_smoothing (float): The label smoothing value.
        reduction (str): The reduction method to apply to the output.
    """

    def __init__(self, label_smoothing=0.0, reduction="mean"):
        """
        Constructor for BCELoss
        """
        super(BCELoss, self).__init__()
        assert (
            0 <= label_smoothing < 1
        ), "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.bce_with_logits = nn.BCELoss(reduction=reduction)

    def forward(self, tensor, target):
        """
        Forward pass for BCELoss
        """
        mask = target != NOCLASS_IDX

        target = target[mask].float()
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = (
                target * positive_smoothed_labels +
                (1 - target) * negative_smoothed_labels
            )

        loss = self.bce_with_logits(tensor[mask], target)
        return loss


class Cross_Entropy(nn.Module):
    """
    Cross Entropy Loss with label smoothing

    Args:
        binary (bool): If True, use BCELoss, otherwise use CrossEntropyLoss.
        weight (Tensor): A manual rescaling weight given to each class.
        label_smoothing (float): The label smoothing value.
        soft (bool): If True, allow soft label from a teacher model.
    """

    def __init__(self, binary, weight=None, label_smoothing=0.1, soft=False):
        super(Cross_Entropy, self).__init__()
        self.binary = binary
        self.soft = soft
        if soft:
            self.loss = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            if self.binary:
                self.loss = BCELoss(label_smoothing=label_smoothing, reduction="mean")
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
