# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

""" Custom schedulers for TAO training workflows. """

import torch


# TODO: @scha to add the schedulder to RTDETR PL module for the next release
class LinearWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear Warmup scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int = 1000,
    ):
        if num_warmup_steps > 0:
            msg = f"num_warmup_steps should be > 0, got {num_warmup_steps}"
            ValueError(msg)
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer, lambda step: min(step / num_warmup_steps, 1.0))
