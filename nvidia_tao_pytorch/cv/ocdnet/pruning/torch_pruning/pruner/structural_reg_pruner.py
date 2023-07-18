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

"""structural regularized pruner module."""
# pylint: disable=W0622
from .. import functional
from typing import Callable
from .basepruner import LocalPruner
import torch


class LocalStructrualRegularizedPruner(LocalPruner):
    """Local Structrual Regularized Pruner class."""

    def __init__(
        self,
        model,
        example_inputs,
        importance,
        total_steps=1,
        beta=1e-4,
        pruning_rate_scheduler: Callable = None,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):
        """Initialize."""
        super(LocalStructrualRegularizedPruner, self).__init__(
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
        self.dropout_groups = {}
        self.beta = beta
        self.plans = self.get_all_plans()

    def estimate_importance(self, plan):
        """estimate importance."""
        return self.importance(plan)

    def structrual_dropout(self, module, input, output):
        """structrual dropout."""
        return self.dropout_groups[module][0](output)

    def regularize(self, model):
        """regularize."""
        for plan in self.plans:
            for dep, _ in plan:
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    functional.prune_conv_out_channel,
                    functional.prune_linear_out_channel,
                ]:
                    # regularize output channels
                    layer.weight.grad.data.add_(self.beta * torch.sign(layer.weight.data))
                elif prune_fn in [
                    functional.prune_conv_in_channel,
                    functional.prune_linear_in_channel,
                ]:
                    # regularize input channels
                    layer.weight.grad.data.add_(self.beta * torch.sign(layer.weight.data))
                elif prune_fn == functional.prune_batchnorm:
                    # regularize BN
                    if layer.affine is not None:
                        layer.weight.grad.data.add_(self.beta * torch.sign(layer.weight.data))
