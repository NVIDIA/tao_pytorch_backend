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

"""Contrastive loss function"""

import torch


class ContrastiveLoss(torch.nn.Module):
    """Contrastive Loss for comparing image embeddings.

    Args:
        margin (float): The margin used for contrastive loss.
    """

    def __init__(self, margin=2.0):
        """Initialize"""
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):
        """
        Compute the contrastive loss.

        Args:
            euclidean_distance (torch.Tensor): Euclidean distance between the two output tensors from the model
            label (torch.Tensor): Label indicating if the images are similar or dissimilar.

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive
