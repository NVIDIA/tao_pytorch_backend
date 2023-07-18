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

"""Metric module."""
# pylint: disable=R1705
import torch
from .dependency import TORCH_CONV, TORCH_BATCHNORM, TORCH_PRELU, TORCH_LINEAR


def norm(weights, p=1, norm_dim=0, idxs=None, reduction='sum'):
    """Norm."""
    l1_norm = torch.norm(weights.transpose(0, norm_dim).flatten(1), p=p, dim=1)
    if idxs is not None:
        l1_norm = l1_norm[idxs]
    if reduction == 'sum':
        return l1_norm.sum()
    return l1_norm


class NormMetric:
    """Norm Metric class."""

    def __init__(self, p, reduction='sum'):
        """Initilize."""
        self.p = p
        self.reduction = reduction

    @torch.no_grad()
    def __call__(self, layer, idxs):
        """Call function."""
        if isinstance(layer, (TORCH_CONV, TORCH_LINEAR)):
            weight_norm = norm(layer.weight, p=self.p, norm_dim=0, idxs=idxs, reduction=self.reduction)
            if layer.bias is not None:
                weight_norm += norm(layer.bias.unsqueeze(-1), p=self.p, norm_dim=0, idxs=idxs, reduction=self.reduction)
            return weight_norm
        elif isinstance(layer, TORCH_BATCHNORM):
            if layer.weight is not None:
                weight_norm = norm(layer.weight.unsqueeze(-1), p=self.p, norm_dim=0, idxs=idxs, reduction=self.reduction) \
                    + norm(layer.bias.unsqueeze(-1), p=self.p, norm_dim=0, idxs=idxs, reduction=self.reduction)
            else:
                weight_norm = 0
            return weight_norm
        elif isinstance(layer, TORCH_PRELU):
            if layer.num_parameters == 1:
                return 0
            else:
                return norm(layer.weight.unsqueeze(-1), p=self.p, norm_dim=0, idxs=idxs, reduction=self.reduction)
        else:
            raise NotImplementedError()
