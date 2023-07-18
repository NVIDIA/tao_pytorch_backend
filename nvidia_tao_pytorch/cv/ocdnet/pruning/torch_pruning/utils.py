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

"""Utility module."""
# pylint: disable=R1705
from .dependency import TORCH_CONV, TORCH_BATCHNORM, TORCH_PRELU, TORCH_LINEAR
import torch
import thop


def count_prunable_params_of_modules(module):
    """Count prunable params of moduls."""
    if isinstance(module, (TORCH_CONV, TORCH_LINEAR)):
        num_params = module.weight.numel()
        if module.bias is not None:
            num_params += module.bias.numel()
        return num_params
    elif isinstance(module, TORCH_BATCHNORM):
        num_params = module.running_mean.numel() + module.running_var.numel()
        if module.affine:
            num_params += module.weight.numel() + module.bias.numel()
        return num_params
    elif isinstance(module, TORCH_PRELU):
        if len(module.weight) == 1:
            return 0
        else:
            return module.weight.numel
    else:
        return 0


def count_prunable_in_channels(module):
    """Count prunable in-channels."""
    if isinstance(module, TORCH_CONV):
        return module.weight.shape[1]
    elif isinstance(module, TORCH_LINEAR):
        return module.in_features
    elif isinstance(module, TORCH_BATCHNORM):
        return module.num_features
    elif isinstance(module, TORCH_PRELU):
        if len(module.weight) == 1:
            return 0
        else:
            return len(module.weight)
    else:
        return 0


def count_prunable_out_channels(module):
    """Count prunable out-channels."""
    if isinstance(module, TORCH_CONV):
        return module.weight.shape[0]
    elif isinstance(module, TORCH_LINEAR):
        return module.out_features
    elif isinstance(module, TORCH_BATCHNORM):
        return module.num_features
    elif isinstance(module, TORCH_PRELU):
        if len(module.weight) == 1:
            return 0
        else:
            return len(module.weight)
    else:
        return 0


def count_params(module):
    """Count params"""
    return sum([p.numel() for p in module.parameters()])


def count_macs_and_params(model, input_size, example_inputs=None):
    """Count macs and params."""
    if example_inputs is None:
        example_inputs = torch.randn(*input_size)
    macs, params = thop.profile(model, inputs=(example_inputs, ), verbose=False)
    return macs, params


def count_total_prunable_channels(model):
    """Count total prunable channels."""
    in_ch = 0
    out_ch = 0
    for m in model.modules():
        out_ch += count_prunable_out_channels(m)
        in_ch += count_prunable_in_channels(m)

    return out_ch, in_ch
