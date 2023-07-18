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

"""Structural dropout pruner."""
# pylint: disable=W0622
from typing import Callable
from .basepruner import LocalPruner
import torch
import torch.nn as nn


def imp_to_prob(x, scale=1.0):
    """Importance to prob."""
    return torch.nn.functional.sigmoid((x - x.mean()) / (x.std() + 1e-8) * scale)


class StructrualDropout(nn.Module):
    """Structual Dropout class."""

    def __init__(self, p):
        """Initialize."""
        super(StructrualDropout, self).__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        """Forward."""
        C = x.shape[1]
        if self.mask is None:
            self.mask = (torch.cuda.FloatTensor(C, device=x.device).uniform_() > self.p).view(1, -1, 1, 1)
        res = x * self.mask
        return res

    def reset(self, p):
        """Reset."""
        self.p = p
        self.mask = None


class StructrualDropoutPruner(LocalPruner):
    """Structual Dropout Pruner class."""

    def __init__(
        self,
        model,
        example_inputs,
        importance,
        total_steps=1,
        p=0.1,
        pruning_rate_scheduler: Callable = None,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):
        """Initialize."""
        super(StructrualDropoutPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            total_steps=total_steps,
            pruning_rate_scheduler=pruning_rate_scheduler,
            ch_sparsity=ch_sparsity,
            layer_ch_sparsity=layer_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            user_defined_parameters=user_defined_parameters,
            output_transform=output_transform,
        )
        self.importance = importance
        self.module2dropout = {}
        self.p = p
        self.plans = self.get_all_plans()

    def estimate_importance(self, plan):
        """Estimate importance."""
        return self.importance(plan)

    def structrual_dropout(self, module, input, output):
        """Structrual Dropout."""
        return self.module2dropout[module][0](output)

    def regularize(self, model):
        """Regularize."""
        pass

    def register_structural_dropout(self, module):
        """Register Structural Dropout."""
        for plan in self.plans:
            dropout_layer = StructrualDropout(p=self.p)
            for dep, _ in plan:
                module = dep.target.module
                if self.ignored_layers is not None and module in self.ignored_layers:
                    continue
                if module in self.module2dropout:
                    continue
                if dep.handler not in self.DG.out_channel_pruners:
                    continue
                hook = module.register_forward_hook(self.structrual_dropout)
                self.module2dropout[module] = (dropout_layer, hook)

    def remove_structural_dropout(self):
        """Remove Structural Dropout."""
        for __, (_, hook) in self.module2dropout.items():
            hook.remove()
