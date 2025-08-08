# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the MIT license.
# Modified by: Daquan Zhou

"""ConvNeXt backbone for FAN hybrid model."""

from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import ClassifierHead, DropPath, Mlp, SelectAdaptivePool2d, trunc_normal_
from timm.models import named_apply, register_notrace_module


def _is_contiguous(tensor: torch.Tensor) -> bool:
    """Check if the tensor is continguous for torch jit script purpose"""
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    return tensor.is_contiguous(memory_format=torch.contiguous_format)


class ConvMlp(nn.Module):
    """MLP using 1x1 convs that keeps spatial dims"""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=None, drop=0.0
    ):
        """Initialize the ConvMlp Class.

        Args:
            in_features: number of input features
            hidden_feautres: number of hidden features
            out_features: number of output features
            act_layer: activation layer class to be used
            norm_layer: normalization layer class to be used
            drop: dropout probability
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward function"""
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


@register_notrace_module
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W)."""

    def __init__(self, normalized_shape, eps=1e-6):
        """Initialize the Layernorm2d class.

        Args:
            normalized_shape: shape to be normalized to
            eps: epsilon value for numerically stability
        """
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        """Forward function."""
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
            ).permute(0, 3, 1, 2)

        s, u = torch.var_mean(x, dim=1, keepdim=True)
        x = (x - u) * torch.rsqrt(s + self.eps)
        x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block

    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    """

    def __init__(self, dim, drop_path=0.0, ls_init_value=1e-6, conv_mlp=True, mlp_ratio=4, norm_layer=None):
        """Initialize ConvNext Block.

        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic depth rate. Default: 0.0
            ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """
        super().__init__()
        if not norm_layer:
            norm_layer = partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.use_conv_mlp = conv_mlp
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=nn.GELU)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Forward function."""
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class ConvNeXtStage(nn.Module):
    """ConvNeXt Stage."""

    def __init__(
        self,
        in_chs,
        out_chs,
        stride=2,
        depth=2,
        dp_rates=None,
        ls_init_value=1.0,
        conv_mlp=True,
        norm_layer=None,
        cl_norm_layer=None,
        no_downsample=False,
    ):
        """Initialize ConvNext Stage.

        Args:
            in_chs (int): Number of input channels.
            out_chs (int): Number of output channels.
        """
        super().__init__()

        if in_chs != out_chs or stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=stride, stride=stride if not no_downsample else 1),
            )
        else:
            self.downsample = nn.Identity()

        dp_rates = dp_rates or [0.0] * depth
        self.blocks = nn.Sequential(
            *[
                ConvNeXtBlock(
                    dim=out_chs,
                    drop_path=dp_rates[j],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    norm_layer=norm_layer if conv_mlp else cl_norm_layer,
                )
                for j in range(depth)
            ]
        )

    def forward(self, x):
        """Forward function."""
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXtFANBackbone(nn.Module):
    """ConvNeXt backbone for FAN hybrid model."""

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        output_stride=32,
        patch_size=4,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        ls_init_value=1e-6,
        conv_mlp=True,
        use_head=True,
        head_init_scale=1.0,
        head_norm_first=False,
        norm_layer=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        remove_last_downsample=False,
    ):
        """Initialize the ConvNext Class

        Args:
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
            dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
            drop_rate (float): Head dropout rate
            drop_path_rate (float): Stochastic depth rate. Default: 0.
            ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
            head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        """
        super().__init__()
        assert output_stride == 32
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)
            cl_norm_layer = norm_layer if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        else:
            assert conv_mlp, (
                "If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input"
            )
            cl_norm_layer = norm_layer

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size), norm_layer(dims[0])
        )

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        curr_stride = patch_size
        prev_chs = dims[0]
        stages = []
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(len(depths)):
            stride = 2 if i > 0 else 1
            # FIXME support dilation / output_stride
            curr_stride *= stride
            out_chs = dims[i]
            no_downsample = remove_last_downsample and (i == len(depths) - 1)
            stages.append(
                ConvNeXtStage(
                    prev_chs,
                    out_chs,
                    stride=stride,
                    depth=depths[i],
                    dp_rates=dp_rates[i],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    norm_layer=norm_layer,
                    cl_norm_layer=cl_norm_layer,
                    no_downsample=no_downsample,
                )
            )
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [{"num_chs": prev_chs, "reduction": curr_stride, "module": f"stages.{i}"}]
        self.stages = nn.Sequential(*stages)

        self.num_features = prev_chs

        if head_norm_first:
            # norm -> global pool -> fc ordering, like most other nets (not compat with FB weights)
            self.norm_pre = norm_layer(self.num_features)  # final norm layer, before pooling
            if use_head:
                self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)
        else:
            # pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
            self.norm_pre = nn.Identity()
            if use_head:
                self.head = nn.Sequential(
                    OrderedDict(
                        [
                            ("global_pool", SelectAdaptivePool2d(pool_type=global_pool)),
                            ("norm", norm_layer(self.num_features)),
                            ("flatten", nn.Flatten(1) if global_pool else nn.Identity()),
                            ("drop", nn.Dropout(self.drop_rate)),
                            ("fc", nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()),
                        ]
                    )
                )

        named_apply(partial(self._init_weights, head_init_scale=head_init_scale), self)

    def _init_weights(self, module, name=None, head_init_scale=1.0):
        """Initialize weights"""
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)
            if name and "head." in name:
                module.weight.data.mul_(head_init_scale)
                module.bias.data.mul_(head_init_scale)

    def get_classifier(self):
        """Returns classifier of ConvNeXt"""
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool="avg"):
        """Redefine the classification head"""
        if isinstance(self.head, ClassifierHead):
            # norm -> global pool -> fc
            self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)
        else:
            # pool -> norm -> fc
            self.head = nn.Sequential(
                OrderedDict(
                    [
                        ("global_pool", SelectAdaptivePool2d(pool_type=global_pool)),
                        ("norm", self.head.norm),
                        ("flatten", nn.Flatten(1) if global_pool else nn.Identity()),
                        ("drop", nn.Dropout(self.drop_rate)),
                        ("fc", nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()),
                    ]
                )
            )

    def forward_features(self, x, return_feat=False):
        """Extract features"""
        x = self.stem(x)
        out_list = []
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            out_list.append(x)
        x = self.norm_pre(x)

        return x, out_list if return_feat else x

    def forward(self, x):
        """Forward function"""
        x = self.forward_features(x)
        x = self.head(x)
        return x
