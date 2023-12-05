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

""" FasterViT Model Module """

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, LayerNorm2d

from mmcls.models.builder import BACKBONES
from mmcv.runner import BaseModule

from nvidia_tao_pytorch.cv.backbone.faster_vit import PatchEmbed, FasterViTLayer


class FasterViT(BaseModule):
    """
    FasterViT based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention <https://arxiv.org/abs/2306.06189>"
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 ct_size,
                 mlp_ratio,
                 num_heads,
                 resolution=224,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 layer_norm_last=False,
                 hat=[False, False, True, False],
                 do_propagation=False,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            in_dim: inner-plane feature size dimension.
            depths: layer depth.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window.
            mlp_ratio: MLP ratio.
            num_heads: number of attention head.
            resolution: image resolution.
            drop_path_rate: drop path rate.
            in_chans: input channel dimension.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            layer_scale: layer scale coefficient.
            layer_scale_conv: conv layer scale coefficient.
            layer_norm_last: last stage layer norm flag.
            hat: hierarchical attention flag.
            do_propagation: enable carrier token propagation.
        """
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        if hat is None:
            hat = [True, ] * len(depths)
        for i in range(len(depths)):
            conv = bool(i in (0, 1))
            level = FasterViTLayer(dim=int(dim * 2 ** i),
                                   depth=depths[i],
                                   num_heads=num_heads[i],
                                   window_size=window_size[i],
                                   ct_size=ct_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   conv=conv,
                                   drop=drop_rate,
                                   attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                   downsample=(i < 3),
                                   layer_scale=layer_scale,
                                   layer_scale_conv=layer_scale_conv,
                                   input_resolution=int(2 ** (-2 - i) * resolution),
                                   only_local=not hat[i],
                                   do_propagation=do_propagation)
            self.levels.append(level)
        self.norm = LayerNorm2d(num_features) if layer_norm_last else nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Returns eywords to ignore during weight decay"""
        return {'rpb'}

    def forward_features(self, x):
        """Extract features"""
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        """Forward function."""
        x = self.forward_features(x)
        return x


@BACKBONES.register_module()
class faster_vit_0_224(FasterViT):
    """FasterViT-0 model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[2, 3, 6, 5],
                            num_heads=[2, 4, 8, 16],
                            window_size=[7, 7, 7, 7],
                            ct_size=2,
                            dim=64,
                            in_dim=64,
                            mlp_ratio=4,
                            resolution=224,
                            drop_path_rate=0.2,
                            hat=[False, False, True, False],
                            **kwargs)
        super(faster_vit_0_224, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class faster_vit_1_224(FasterViT):
    """FasterViT-1 model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[1, 3, 8, 5],
                            num_heads=[2, 4, 8, 16],
                            window_size=[7, 7, 7, 7],
                            ct_size=2,
                            dim=80,
                            in_dim=32,
                            mlp_ratio=4,
                            resolution=224,
                            drop_path_rate=0.2,
                            hat=[False, False, True, False],
                            **kwargs)
        super(faster_vit_1_224, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class faster_vit_2_224(FasterViT):
    """FasterViT-2 model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 3, 8, 5],
                            num_heads=[2, 4, 8, 16],
                            window_size=[7, 7, 7, 7],
                            ct_size=2,
                            dim=96,
                            in_dim=64,
                            mlp_ratio=4,
                            resolution=224,
                            drop_path_rate=0.2,
                            hat=[False, False, True, False],
                            **kwargs)
        super(faster_vit_2_224, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class faster_vit_3_224(FasterViT):
    """FasterViT-3 model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 3, 12, 5],
                            num_heads=[2, 4, 8, 16],
                            window_size=[7, 7, 7, 7],
                            ct_size=2,
                            dim=128,
                            in_dim=64,
                            mlp_ratio=4,
                            resolution=224,
                            drop_path_rate=0.3,
                            layer_scale=1e-5,
                            hat=[False, False, True, False],
                            **kwargs)
        super(faster_vit_3_224, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class faster_vit_4_224(FasterViT):
    """FasterViT-4 model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 3, 12, 5],
                            num_heads=[4, 8, 16, 32],
                            window_size=[7, 7, 7, 7],
                            ct_size=2,
                            dim=196,
                            in_dim=64,
                            mlp_ratio=4,
                            resolution=224,
                            drop_path_rate=0.3,
                            layer_scale=1e-5,
                            hat=[False, False, True, False],
                            **kwargs)
        super(faster_vit_4_224, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class faster_vit_5_224(FasterViT):
    """FasterViT-5 model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 3, 12, 5],
                            num_heads=[4, 8, 16, 32],
                            window_size=[7, 7, 7, 7],
                            ct_size=2,
                            dim=320,
                            in_dim=64,
                            mlp_ratio=4,
                            resolution=224,
                            drop_path_rate=0.3,
                            layer_scale=1e-5,
                            hat=[False, False, True, False],
                            **kwargs)
        super(faster_vit_5_224, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class faster_vit_6_224(FasterViT):
    """FasterViT-6 model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 3, 16, 8],
                            num_heads=[4, 8, 16, 32],
                            window_size=[7, 7, 7, 7],
                            ct_size=2,
                            dim=320,
                            in_dim=64,
                            mlp_ratio=4,
                            resolution=224,
                            drop_path_rate=0.5,
                            layer_scale=1e-5,
                            hat=[False, False, True, False],
                            **kwargs)
        super(faster_vit_6_224, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class faster_vit_4_21k_224(FasterViT):
    """FasterViT-4-21k-224 model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 3, 12, 5],
                            num_heads=[4, 8, 16, 32],
                            window_size=[7, 7, 14, 7],
                            ct_size=2,
                            dim=196,
                            in_dim=64,
                            mlp_ratio=4,
                            resolution=224,
                            drop_path_rate=0.42,
                            layer_scale=1e-5,
                            hat=[False, False, False, False],
                            **kwargs)
        super(faster_vit_4_21k_224, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class faster_vit_4_21k_384(FasterViT):
    """FasterViT-4-21k-384 model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 3, 12, 5],
                            num_heads=[4, 8, 16, 32],
                            window_size=[7, 7, 24, 12],
                            ct_size=2,
                            dim=196,
                            in_dim=64,
                            mlp_ratio=4,
                            resolution=384,
                            drop_path_rate=0.42,
                            layer_scale=1e-5,
                            hat=[False, False, False, False],
                            **kwargs)
        super(faster_vit_4_21k_384, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class faster_vit_4_21k_512(FasterViT):
    """FasterViT-4-21k-512 model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 3, 12, 5],
                            num_heads=[4, 8, 16, 32],
                            window_size=[7, 7, 32, 16],
                            ct_size=2,
                            dim=196,
                            in_dim=64,
                            mlp_ratio=4,
                            resolution=512,
                            drop_path_rate=0.42,
                            layer_scale=1e-5,
                            hat=[False, False, False, False],
                            **kwargs)
        super(faster_vit_4_21k_512, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class faster_vit_4_21k_768(FasterViT):
    """FasterViT-4-21k-768 model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 3, 12, 5],
                            num_heads=[4, 8, 16, 32],
                            window_size=[7, 7, 48, 24],
                            ct_size=2,
                            dim=196,
                            in_dim=64,
                            mlp_ratio=4,
                            resolution=768,
                            drop_path_rate=0.42,
                            layer_scale=1e-5,
                            hat=[False, False, False, False],
                            **kwargs)
        super(faster_vit_4_21k_768, self).__init__(**model_kwargs)
