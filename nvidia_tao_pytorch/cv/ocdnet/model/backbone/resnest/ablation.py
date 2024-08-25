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
#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
"""ResNeSt ablation study models"""

from .resnet import ResNet, Bottleneck

__all__ = ['resnest50_fast_1s1x64d', 'resnest50_fast_2s1x64d', 'resnest50_fast_4s1x64d',
           'resnest50_fast_1s2x40d', 'resnest50_fast_2s2x40d', 'resnest50_fast_4s2x40d',
           'resnest50_fast_1s4x24d']


def resnest50_fast_1s1x64d(pretrained=False, root='~/.encoding/models', **kwargs):
    """Resnest50_fast_1s1x64d model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=1, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    return model


def resnest50_fast_2s1x64d(pretrained=False, root='~/.encoding/models', **kwargs):
    """Resnest50_fast_2s1x64d model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    return model


def resnest50_fast_4s1x64d(pretrained=False, root='~/.encoding/models', **kwargs):
    """Resnest50_fast_4s1x64d model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=4, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    return model


def resnest50_fast_1s2x40d(pretrained=False, root='~/.encoding/models', **kwargs):
    """Resnest50_fast_1s2x40d model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=1, groups=2, bottleneck_width=40,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    return model


def resnest50_fast_2s2x40d(pretrained=False, root='~/.encoding/models', **kwargs):
    """Resnest50_fast_2s2x40d model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=2, bottleneck_width=40,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    return model


def resnest50_fast_4s2x40d(pretrained=False, root='~/.encoding/models', **kwargs):
    """Resnest50_fast_4s2x40d model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=4, groups=2, bottleneck_width=40,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    return model


def resnest50_fast_1s4x24d(pretrained=False, root='~/.encoding/models', **kwargs):
    """Resnest50_fast_1s4x24d model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=1, groups=4, bottleneck_width=24,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    return model
