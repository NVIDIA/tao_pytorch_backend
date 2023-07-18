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

"""Strategy module."""
import torch
from abc import abstractclassmethod, ABC
from typing import Sequence
import random


def round_pruning_amount(total_parameters, n_to_prune, round_to):
    """round the parameter amount after pruning to an integer multiple of `round_to`."""
    n_remain = round_to * max(int(total_parameters - n_to_prune) // round_to, 1)
    return max(total_parameters - n_remain, 0)


class BaseStrategy(ABC):
    """Base Strategy class."""

    def __call__(self, *args, **kwargs):
        """Use Base Strategy."""
        return self.apply(*args, **kwargs)

    @abstractclassmethod
    def apply(cls, weights, amount=0.0, round_to=1) -> Sequence[int]:
        """ Apply the strategy on weights with user specified pruning percentage.

        Args:
            amount (float): the percentage of weights to be pruned (amount<1.0) or the amount of weights to be pruned (amount>=1.0)
            round_to (int): the number to which the number of pruned channels is rounded.
        """
        raise NotImplementedError


class RandomStrategy(BaseStrategy):
    """Random Strategy class."""

    def apply(self, weights, amount=0.0, round_to=1) -> Sequence[int]:
        """Return indices."""
        if amount <= 0:
            return []
        n = len(weights)
        n_to_prune = int(amount * n) if amount < 1.0 else amount
        n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
        if n_to_prune == 0:
            return []
        indices = random.sample(list(range(n)), k=n_to_prune)
        return indices


class LNStrategy(BaseStrategy):
    """LNStrategy class."""

    def __init__(self, p):
        """Initialize."""
        self.p = p

    def apply(self, weights, amount=0.0, round_to=1) -> Sequence[int]:
        """Return indices."""
        if amount <= 0:
            return []
        n = len(weights)
        l1_norm = torch.norm(weights.view(n, -1), p=self.p, dim=1)
        n_to_prune = int(amount * n) if amount < 1.0 else amount
        n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
        if n_to_prune == 0:
            return []
        threshold = torch.kthvalue(l1_norm, k=n_to_prune).values
        indices = torch.nonzero(l1_norm <= threshold).view(-1).tolist()
        return indices


class L1Strategy(LNStrategy):
    """L1Strategy class."""

    def __init__(self):
        """Initialize."""
        super(L1Strategy, self).__init__(p=1)


class L2Strategy(LNStrategy):
    """L2Strategy class."""

    def __init__(self):
        """Initialize."""
        super(L2Strategy, self).__init__(p=2)


class GroupLNStrategy(ABC):
    """GroupLNStrategy class."""

    def __call__(self, *args, **kwargs):
        """Call function."""
        return self.apply(*args, **kwargs)

    @abstractclassmethod
    def apply(cls, group, amount=0.0, round_to=1) -> Sequence[int]:
        """ Apply the strategy on weights with user specified pruning percentage.

        Args:
            amount (float): the percentage of weights to be pruned (amount<1.0) or the amount of weights to be pruned (amount>=1.0)
            round_to (int): the number to which the number of pruned channels is rounded.
        """
        for dep, idxs in cls._plans:
            _, __ = dep.handler(dep.target.module, idxs, dry_run=True)
