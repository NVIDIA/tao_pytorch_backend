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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Swin Transformer backbone."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_
from timm.models.vision_transformer import init_weights_vit_timm

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase
from nvidia_tao_pytorch.cv.backbone_v2.swin_utils import (
    BasicLayer,
    PatchMerging,
    PatchEmbed,
)


class SwinTransformer(BackboneBase):
    """Swin Transformer using FAN blocks.

    Swin Transformer (the name Swin stands for Shifted window) serves as a general-purpose backbone for computer
    vision. It is basically a hierarchical Transformer whose representation is computed with shifted windows. The
    shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local
    windows while also allowing for cross-window connection.

    References:
    - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
    - [https://github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        weight_init="",
        mlp_type="Mlp",
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        out_indices=(0, 1, 2, 3),
        dilation=False,
        post_norm=False,
        **kwargs,
    ):
        """Initialize the SwinTransformer model.

        Args:
            img_size (int | tuple(int)): Input image size. Default 224
            patch_size (int | tuple(int)): Patch size. Default: 4
            in_chans (int): Number of input image channels. Default: `3`.
            num_classes (int): Number of classes for classification head. Default: `1000`.
            embed_dim (int): Patch embedding dimension. Default: 96
            depths (tuple(int)): Depth of each Swin Transformer layer.
            num_heads (tuple(int)): Number of attention heads in different layers.
            window_size (int): Window size. Default: 7
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            drop_rate (float): Dropout rate. Default: 0
            attn_drop_rate (float): Attention dropout rate. Default: 0
            drop_path_rate (float): Stochastic depth rate. Default: 0.1
            norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
            ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
            patch_norm (bool): If True, add normalization after patch embedding. Default: True
            activation_checkpoint (bool): Whether to use activation checkpointing. Default: `False`.
            freeze_at (list): List of keys corresponding to the stages or layers to freeze. If `None`, no specific
                layers are frozen. If `"all"`, the entire model is frozen and set to eval mode. Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the backbone will be frozen. Default: `False`.
        """
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )

        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.dilation = dilation

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                img_size[0] // patch_size[0],
                img_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        # prepare downsample list
        downsamplelist = [PatchMerging for i in range(self.num_layers)]
        downsamplelist[-1] = None
        self.num_inter_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        if self.dilation:
            downsamplelist[-2] = None
            self.num_inter_features[-1] = int(embed_dim * 2 ** (self.num_layers - 1)) // 2
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                # dim=int(embed_dim * 2 ** i_layer),
                dim=self.num_inter_features[i_layer],
                depth=depths[i_layer],
                mlp_type=mlp_type[i_layer] if isinstance(mlp_type, list) else mlp_type,
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[: i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                downsample=downsamplelist[i_layer],
                use_checkpoint=activation_checkpoint,
            )
            self.layers.append(layer)

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(self.num_inter_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        if post_norm:
            self.post_norm = nn.LayerNorm(self.num_features)
        else:
            self.post_norm = None

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        assert weight_init in ("jax", "jax_nlhb", "nlhb", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in weight_init else 0.0
        if weight_init.startswith("jax"):
            for n, m in self.named_modules():
                init_weights_vit_timm(m, n, head_bias=head_bias, jax_impl=True)  # pylint: disable=E1123
        else:
            self.apply(init_weights_vit_timm)

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {}
        # TODO(@yuw, @hongyuc): No stem. Add patch_embed as stage 0?
        for i, layer in enumerate(self.layers, start=1):
            stage_dict[i] = layer
        return stage_dict

    @torch.jit.ignore
    def no_weight_decay(self):
        """Get the set of parameter names to exclude from weight decay."""
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Get the set of parameter keywords to exclude from weight decay."""
        return {"relative_position_bias_table"}

    def get_classifier(self):
        """Get the classifier module."""
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        """Reset the classifier head."""
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """Forward pass through the backbone, excluding the head."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        for layer in self.layers:
            _, _, _, x, Wh, Ww = layer(x, Wh, Ww)
        if self.post_norm:
            x = self.post_norm(x)  # B L C
        return x

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head."""
        x = self.forward_features(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = {}
        for idx, layer in enumerate(self.layers):
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_inter_features[idx]).permute(0, 3, 1, 2).contiguous()
                outs[f'p{idx}'] = out
        return outs

    def forward(self, x):
        """Forward."""
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def swin_tiny_patch4_window7_224(**kwargs):
    """Swin-T @ 224x224, trained ImageNet-1k"""
    return SwinTransformer(
        img_size=(224, 224),
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def swin_small_patch4_window7_224(**kwargs):
    """Swin-S @ 224x224"""
    return SwinTransformer(
        img_size=(224, 224),
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def swin_base_patch4_window7_224(**kwargs):
    """Swin-B @ 224x224"""
    return SwinTransformer(
        img_size=(224, 224),
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def swin_base_patch4_window12_384(**kwargs):
    """Swin-B @ 384x384"""
    return SwinTransformer(
        img_size=(384, 384),
        patch_size=4,
        window_size=12,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def swin_large_patch4_window7_224(**kwargs):
    """Swin-L @ 224x224"""
    return SwinTransformer(
        img_size=(224, 224),
        patch_size=4,
        window_size=7,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def swin_large_patch4_window12_384(**kwargs):
    """Swin-L @ 384x384"""
    return SwinTransformer(
        img_size=(384, 384),
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        **kwargs,
    )
