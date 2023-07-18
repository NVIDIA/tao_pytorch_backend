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

# This file is modified from https://github.com/traveller59/second.pytorch
"""FastAI Learning Rate Scheduler."""
import math
from functools import partial

import numpy as np
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper


class LRSchedulerStep(object):
    """Step LR scheduler"""

    def __init__(self, fai_optimizer: OptimWrapper, total_step, lr_phases,
                 mom_phases):
        """Initialize."""
        # if not isinstance(fai_optimizer, OptimWrapper):
        #     raise TypeError('{} is not a fastai OptimWrapper'.format(
        #         type(fai_optimizer).__name__))
        self.optimizer = fai_optimizer
        self.total_step = total_step
        self.lr_phases = []

        for i, (start, lambda_func) in enumerate(lr_phases):
            if len(self.lr_phases) != 0:
                assert self.lr_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)  # nosec
            if i < len(lr_phases) - 1:
                self.lr_phases.append((int(start * total_step), int(lr_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.lr_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.lr_phases[0][0] == 0
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                assert self.mom_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)  # nosec
            if i < len(mom_phases) - 1:
                self.mom_phases.append((int(start * total_step), int(mom_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.mom_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.mom_phases[0][0] == 0

    def step(self, step):
        """Step."""
        for start, end, func in self.lr_phases:
            if step >= start:
                self.optimizer.lr = func((step - start) / (end - start))
        for start, end, func in self.mom_phases:
            if step >= start:
                self.optimizer.mom = func((step - start) / (end - start))


def annealing_cos(start, end, pct):
    """Cosine Annealing."""
    # print(pct, start, end)
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycle(LRSchedulerStep):
    """One Cycle LR scheduler."""

    def __init__(self, fai_optimizer, total_step, lr_max, moms, div_factor,
                 pct_start):
        """Initialize."""
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        low_lr = self.lr_max / self.div_factor
        lr_phases = ((0, partial(annealing_cos, low_lr, self.lr_max)),
                     (self.pct_start,
                      partial(annealing_cos, self.lr_max, low_lr / 1e4)))
        mom_phases = ((0, partial(annealing_cos, *self.moms)),
                      (self.pct_start, partial(annealing_cos,
                                               *self.moms[::-1])))
        fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0]
        super().__init__(fai_optimizer, total_step, lr_phases, mom_phases)


class CosineWarmupLR(lr_sched._LRScheduler):
    """Cosine warmup LR scheduler."""

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        """Initialize."""
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get learning rate."""
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


class FakeOptim:
    """Fake optimizer."""

    def __init__(self):
        """Initialize."""
        self.lr = 0
        self.mom = 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    opt = FakeOptim()  # 3e-3, wd=0.4, div_factor=10
    schd = OneCycle(opt, 100, 3e-3, (0.95, 0.85), 10.0, 0.1)

    lrs = []
    moms = []
    for i in range(100):
        schd.step(i)
        lrs.append(opt.lr)
        moms.append(opt.mom)
    plt.plot(lrs)
    # plt.plot(moms)
    plt.show()
    plt.plot(moms)
    plt.show()
