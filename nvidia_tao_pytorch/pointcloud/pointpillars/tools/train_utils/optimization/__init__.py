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

"""Training optimization utilities."""
from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def build_optimizer(model, optim_cfg):
    """Build optimizer."""
    if optim_cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    elif optim_cfg.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay,
            momentum=optim_cfg.momentum
        )
    elif optim_cfg.optimizer == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]  # noqa: E731
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]  # noqa: E731

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.weight_decay, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    """Build learning rate scheduler."""
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.decay_step_list]

    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.lr_decay
        return max(cur_decay, optim_cfg.lr_clip / optim_cfg.lr)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.optimizer == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.lr, list(optim_cfg.moms), optim_cfg.div_factor, optim_cfg.pct_start
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.lr_warmup:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.warmup_epoch * len(total_iters_each_epoch),
                eta_min=optim_cfg.lr / optim_cfg.div_factor
            )

    return lr_scheduler, lr_warmup_scheduler
