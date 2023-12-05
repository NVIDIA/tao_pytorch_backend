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

"""Unstructured pruning."""
import torch
from copy import deepcopy


__all__ = ['mask_weight', 'mask_bias']


def _mask_weight_hook(module, inp):
    if hasattr(module, 'weight_mask'):
        module.weight.data *= module.weight_mask


def _mask_bias_hook(module, inp):
    if module.bias is not None and hasattr(module, 'bias_mask'):
        module.bias.data *= module.bias_mask


def mask_weight(layer, mask, inplace=True):
    """Unstructed pruning for convolution layer

    Args:
        layer: a convolution layer.
        mask: 0-1 mask.
    """
    if not inplace:
        layer = deepcopy(layer)
    if mask.shape != layer.weight.shape:
        return layer
    mask = torch.tensor(mask, dtype=layer.weight.dtype, device=layer.weight.device, requires_grad=False)
    if hasattr(layer, 'weight_mask'):
        mask = mask + layer.weight_mask
        mask[mask > 0] = 1
        layer.weight_mask = mask
    else:
        layer.register_buffer('weight_mask', mask)

    layer.register_forward_pre_hook(_mask_weight_hook)
    return layer


def mask_bias(layer, mask, inplace=True):
    """Unstructed pruning for convolution layer

    Args:
        layer: a convolution layer.
        mask: 0-1 mask.
    """
    if not inplace:
        layer = deepcopy(layer)
    if layer.bias is None or mask.shape != layer.bias.shape:
        return layer

    mask = torch.tensor(mask, dtype=layer.weight.dtype, device=layer.weight.device, requires_grad=False)
    if hasattr(layer, 'bias_mask'):
        mask = mask + layer.bias_mask
        mask[mask > 0] = 1
        layer.bias_mask = mask
    else:
        layer.register_buffer('bias_mask', mask)
    layer.register_forward_pre_hook(_mask_bias_hook)
    return layer
