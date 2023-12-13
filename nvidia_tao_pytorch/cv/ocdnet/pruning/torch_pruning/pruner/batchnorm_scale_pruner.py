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

"""Batchnorm scale pruner moduel."""
from typing import Callable
from .basepruner import LocalPruner, GlobalPruner
import torch
import torch.nn as nn


class LocalBNScalePruner(LocalPruner):
    """Local BN Scale Pruner class."""

    def __init__(
        self,
        model,
        example_inputs,
        importance,
        beta=1e-5,
        total_steps=1,
        pruning_rate_scheduler: Callable = None,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):
        """Initialize."""
        super(LocalBNScalePruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            total_steps=total_steps,
            pruning_rate_scheduler=pruning_rate_scheduler,
            ch_sparsity=ch_sparsity,
            layer_ch_sparsity=layer_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            user_defined_parameters=user_defined_parameters,
            output_transform=output_transform,
        )
        self.beta = beta

    def regularize(self, model):
        """regularize."""
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(self.beta * torch.sign(m.weight.data))


class GlobalBNScalePruner(GlobalPruner):
    """Global BN Scale Pruner class."""

    def __init__(
        self,
        model,
        example_inputs,
        importance,
        beta=1e-5,
        total_steps=1,
        pruning_rate_scheduler: Callable = None,
        ch_sparsity=0.5,
        max_ch_sparsity=1.0,
        layer_ch_sparsity=None,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):
        """Initialize."""
        super(GlobalBNScalePruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            total_steps=total_steps,
            pruning_rate_scheduler=pruning_rate_scheduler,
            ch_sparsity=ch_sparsity,
            max_ch_sparsity=max_ch_sparsity,
            layer_ch_sparsity=layer_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            user_defined_parameters=user_defined_parameters,
            output_transform=output_transform,
        )
        self.beta = beta

    def regularize(self, model):
        """regularize."""
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(self.beta * torch.sign(m.weight.data))
