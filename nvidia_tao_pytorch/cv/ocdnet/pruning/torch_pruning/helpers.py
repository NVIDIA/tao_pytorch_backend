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

"""Helper module."""
# pylint: disable=W0612,W0235
import torch
import torch.nn as nn
from . import functional
import numpy as np
from operator import add
from numbers import Number


def is_scalar(x):
    """Check if is scalar."""
    if isinstance(x, torch.Tensor):
        return len(x.shape) == 0
    if isinstance(x, Number):
        return True
    if isinstance(x, (list, tuple)):
        return False
    return False


class _CustomizedOp(nn.Module):
    def __init__(self, op_class):
        self.op_cls = op_class

    def __repr__(self):
        return "CustomizedOp({})".format(str(self.op_cls))


######################################################
# Dummy module
class _ConcatOp(nn.Module):
    def __init__(self):
        super(_ConcatOp, self).__init__()
        self.offsets = None

    def __repr__(self):
        return "_ConcatOp({})".format(self.offsets)


class DummyMHA(nn.Module):
    """DummyMHA class."""

    def __init__(self):
        """Initialize."""
        super(DummyMHA, self).__init__()


class _SplitOp(nn.Module):
    def __init__(self):
        super(_SplitOp, self).__init__()
        self.offsets = None

    def __repr__(self):
        return "_SplitOp({})".format(self.offsets)


class _ElementWiseOp(nn.Module):
    def __init__(self, grad_fn):
        super(_ElementWiseOp, self).__init__()
        self._grad_fn = grad_fn

    def __repr__(self):
        return "_ElementWiseOp({})".format(self._grad_fn)


######################################################
# Dummy Pruning fn
class DummyPruner(functional.BasePruner):
    """Dummy pruning class."""

    def __call__(self, layer, *args, **kargs):
        """Call function."""
        return layer, {}

    def calc_nparams_to_prune(self, layer, idxs):
        """Calculate nparams to prune."""
        return 0

    def prune(self, layer, idxs):
        """Pruning."""
        return layer


class ConcatPruner(DummyPruner):
    """ConcatPruner class."""

    pass


class SplitPruner(DummyPruner):
    """SplitPruner class."""

    pass


class ElementWiseOpPruner(DummyPruner):
    """ElementWiseOp Pruner class."""

    pass


_prune_concat = ConcatPruner()
_prune_split = SplitPruner()
_prune_elementwise_op = ElementWiseOpPruner()


######################################################
# Index transform
class _FlattenIndexTransform(object):
    def __init__(self, stride=1, reverse=False):
        self._stride = stride
        self.reverse = reverse

    def __call__(self, idxs):
        new_idxs = []
        if self.reverse is True:
            for i in idxs:
                new_idxs.append(i // self._stride)
                new_idxs = list(set(new_idxs))
        else:
            for i in idxs:
                new_idxs.extend(list(range(i * self._stride, (i + 1) * self._stride)))
        return new_idxs


class _ConcatIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):

        if self.reverse is True:
            new_idxs = [
                i - self.offset[0]
                for i in idxs
                if (self.offset[0] <= i < self.offset[1])
            ]
        else:
            new_idxs = [i + self.offset[0] for i in idxs]
        return new_idxs


class _SplitIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse is True:
            new_idxs = [i + self.offset[0] for i in idxs]
        else:
            new_idxs = [
                i - self.offset[0]
                for i in idxs
                if (self.offset[0] <= i < self.offset[1])
            ]
        return new_idxs


class _GroupConvIndexTransform(object):
    def __init__(self, in_channels, out_channels, groups, reverse=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse is True:
            new_idxs = [i + self.offset[0] for i in idxs]
        else:
            group_histgram = np.histogram(  # noqa: F841
                idxs, bins=self.groups, range=(0, self.out_channels)
            )
        return new_idxs


class GConv(nn.Module):
    """GConv class."""

    def __init__(self, gconv):
        """Initialize."""
        super(GConv, self).__init__()
        self.groups = gconv.groups
        self.convs = nn.ModuleList()
        oc_size = gconv.out_channels // self.groups
        ic_size = gconv.in_channels // self.groups
        for _ in range(self.groups):
            self.convs.append(
                nn.Conv2d(
                    in_channels=oc_size,
                    out_channels=ic_size,
                    kernel_size=gconv.kernel_size,
                    stride=gconv.stride,
                    padding=gconv.padding,
                    dilation=gconv.dilation,
                    groups=1,
                    bias=gconv.bias is not None,
                    padding_mode=gconv.padding_mode,
                )
            )
        # copy parameters
        gconv_weight = gconv.weight
        for (i, conv) in enumerate(self.convs):
            conv.weight.data = gconv_weight.data[oc_size * i: oc_size * (i + 1)]
            if gconv.bias is not None:
                conv.bias.data = gconv.bias.data[oc_size * i: oc_size * (i + 1)]

    def forward(self, x):
        """Forward."""
        split_sizes = [conv.in_channels for conv in self.convs]
        xs = torch.split(x, split_sizes, dim=1)
        out = torch.cat([conv(xi) for (conv, xi) in zip(self.convs, xs)], dim=1)
        return out


def gconv2convs(module):
    """GConv to convs."""
    new_module = module
    if (
        isinstance(module, nn.Conv2d) and
        module.groups > 1 and
        module.groups != module.in_channels
    ):
        new_module = GConv(module)
    for name, child in module.named_children():
        new_module.add_module(name, gconv2convs(child))
    return new_module


class ScalarSum:
    """ScalarSum class."""

    def __init__(self):
        """Initialize."""
        self._results = {}

    def update(self, metric_name, metric_value):
        """Update."""
        if metric_name not in self._results:
            self._results[metric_name] = 0
        self._results[metric_name] += metric_value

    def results(self):
        """Return results."""
        return self._results

    def reset(self):
        """Reset."""
        self._results = {}


class VectorSum:
    """VectorSum class."""

    def __init__(self):
        """Initialize."""
        self._results = {}

    def update(self, metric_name, metric_value):
        """Update."""
        if metric_name not in self._results:
            self._results[metric_name] = metric_value
        if isinstance(metric_value, torch.Tensor):
            self._results[metric_name] += metric_value
        elif isinstance(metric_value, list):
            self._results[metric_name] = list(
                map(add, self._results[metric_name], metric_value)
            )

    def results(self):
        """Return results."""
        return self._results

    def reset(self):
        """Reset."""
        self._results = {}
