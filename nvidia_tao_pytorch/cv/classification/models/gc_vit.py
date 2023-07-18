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

""" GCViT Model Module """

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from mmcls.models.builder import BACKBONES
from mmcv.runner import BaseModule

from nvidia_tao_pytorch.cv.backbone.gc_vit import PatchEmbed, GCViTLayer, _to_channel_first


class GCViT(BaseModule):
    """
    GCViT based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 depths,
                 window_size,
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
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 use_rel_pos_bias=True,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            resolution: input image resolution.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            use_rel_pos_bias: set bias for relative positional embedding
        """
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, dim=dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            level = GCViTLayer(dim=int(dim * 2 ** i),
                               depth=depths[i],
                               num_heads=num_heads[i],
                               window_size=window_size[i],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               norm_layer=norm_layer,
                               downsample=(i < len(depths) - 1),
                               layer_scale=layer_scale,
                               input_resolution=int(2 ** (-2 - i) * resolution),
                               image_resolution=resolution,
                               use_rel_pos_bias=use_rel_pos_bias)
            self.levels.append(level)
        self.norm = norm_layer(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Returns eywords to ignore during weight decay"""
        return {'rpb'}

    def forward_features(self, x):
        """Extract features"""
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.norm(x)
        x = _to_channel_first(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        """Forward function."""
        x = self.forward_features(x)
        return x


@BACKBONES.register_module()
class gc_vit_xxtiny(GCViT):
    """GCViT-XXTiny model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[2, 2, 6, 2],
                            num_heads=[2, 4, 8, 16],
                            window_size=[7, 7, 14, 7],
                            dim=64,
                            mlp_ratio=3,
                            drop_path_rate=0.2,
                            **kwargs)
        super(gc_vit_xxtiny, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class gc_vit_xtiny(GCViT):
    """GCViT-XTiny model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 4, 6, 5],
                            num_heads=[2, 4, 8, 16],
                            window_size=[7, 7, 14, 7],
                            dim=64,
                            mlp_ratio=3,
                            drop_path_rate=0.2,
                            **kwargs)
        super(gc_vit_xtiny, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class gc_vit_tiny(GCViT):
    """GCViT-Tiny model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 4, 19, 5],
                            num_heads=[2, 4, 8, 16],
                            window_size=[7, 7, 14, 7],
                            dim=64,
                            mlp_ratio=3,
                            drop_path_rate=0.2,
                            **kwargs)
        super(gc_vit_tiny, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class gc_vit_small(GCViT):
    """GCViT-Small model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 4, 19, 5],
                            num_heads=[3, 6, 12, 24],
                            window_size=[7, 7, 14, 7],
                            dim=96,
                            mlp_ratio=2,
                            drop_path_rate=0.3,
                            layer_scale=1e-5,
                            **kwargs)
        super(gc_vit_small, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class gc_vit_base(GCViT):
    """GCViT-Base model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 4, 19, 5],
                            num_heads=[4, 8, 16, 32],
                            window_size=[7, 7, 14, 7],
                            dim=128,
                            mlp_ratio=2,
                            drop_path_rate=0.5,
                            layer_scale=1e-5,
                            **kwargs)
        super(gc_vit_base, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class gc_vit_large(GCViT):
    """GCViT-Large model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 4, 19, 5],
                            num_heads=[6, 12, 24, 48],
                            window_size=[7, 7, 14, 7],
                            dim=192,
                            mlp_ratio=2,
                            drop_path_rate=0.5,
                            layer_scale=1e-5,
                            **kwargs)
        super(gc_vit_large, self).__init__(**model_kwargs)


@BACKBONES.register_module()
class gc_vit_large_384(GCViT):
    """GCViT-Large Input Resolution 384 model."""

    def __init__(self, **kwargs):
        """Initialize"""
        model_kwargs = dict(depths=[3, 4, 19, 5],
                            num_heads=[6, 12, 24, 48],
                            window_size=[12, 12, 24, 12],
                            dim=192,
                            mlp_ratio=2,
                            drop_path_rate=0.5,
                            layer_scale=1e-5,
                            **kwargs)
        super(gc_vit_large_384, self).__init__(**model_kwargs)
