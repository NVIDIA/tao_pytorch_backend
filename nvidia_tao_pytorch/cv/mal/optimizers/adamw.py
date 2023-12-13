# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MAL/blob/main/LICENSE
"""AdamW optimizer with step."""

from torch.optim import AdamW


class AdamWwStep(AdamW):
    """AdamW optimizer with step."""

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

        for param_group in self.param_groups:
            param_group['step'] = 0
            param_group['epoch'] = 0

    def step(self, closure=None):
        """Step."""
        super().step(closure)
        for param_group in self.param_groups:
            param_group['step'] = param_group['step'] + 1

    def next_epoch(self):
        """Next epoch."""
        for param_group in self.param_groups:
            param_group['epoch'] += 1
