# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""FAN Backbone for RT-DETR"""

from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.layers import trunc_normal_, to_2tuple

from nvidia_tao_pytorch.cv.backbone.convnext_utils import _create_hybrid_backbone
from nvidia_tao_pytorch.cv.backbone.fan import PositionalEncodingFourier, ConvPatchEmbed, ClassAttentionBlock
from nvidia_tao_pytorch.cv.dino.model.fan import HybridEmbed, FANBlock


class FAN(nn.Module):
    """FAN implementation from https://arxiv.org/abs/2204.12451
    Based on timm https://github.com/rwightman/pytorch-image-models/tree/master/timm
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., backbone=None,
                 act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=False,
                 out_index=-1, out_channels=None, out_indices=[0, 1, 2, 3], patch_embed="ConvNext", activation_checkpoint=True):
        """Initialize FAN class"""
        super(FAN, self).__init__()
        img_size = to_2tuple(img_size)

        self.activation_checkpoint = activation_checkpoint
        self.out_index = out_index
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.out_channels = out_channels
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        if patch_embed == "ConvNext":
            self.patch_embed = HybridEmbed(backbone=backbone, patch_size=2, embed_dim=embed_dim)
        else:
            self.patch_embed = ConvPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, act_layer=act_layer)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        build_block = FANBlock
        self.blocks = nn.ModuleList([
            build_block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer, eta=eta)
            for _ in range(depth)])

        self.cls_attn_blocks = nn.ModuleList([
            ClassAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, act_layer=act_layer, norm_layer=norm_layer, eta=eta, tokens_norm=tokens_norm)
            for _ in range(cls_attn_layers)])

        self.out_indices = out_indices

        for i_layer in self.out_indices:
            layer = nn.LayerNorm(self.out_channels[i_layer])
            layer_name = f'out_norm{i_layer}'
            self.add_module(layer_name, layer)

        self.learnable_downsample = nn.Conv2d(in_channels=embed_dim,
                                              out_channels=768,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              dilation=1,
                                              groups=1,
                                              bias=True)

        # Init weights
        trunc_normal_(self.cls_token, std=.02)
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
    def no_weight_decay(self):
        """layers to ignore for weight decay"""
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        """Returns classifier"""
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        """Redefine classifier of FAN"""
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """Extract features
        Args:
            x: tensor

        Returns:
            final_outs: dictionary containing indice name as key and indice feature as value
        """
        outs = []
        B = x.shape[0]
        if isinstance(self.patch_embed, HybridEmbed):
            x, (Hp, Wp), out_list = self.patch_embed(x, return_feat=True)
            outs = outs + out_list
            out_index = [self.out_index]
        else:
            x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos_embed:
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            blk.H, blk.W = Hp, Wp
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = blk(x)
            else:
                x = checkpoint.checkpoint(blk, x)

            Hp, Wp = blk.H, blk.W

            if idx in out_index:
                outs.append(x.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous())
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.cls_attn_blocks:
            x = blk(x)

        x = x[:, 1:, :].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()
        x = self.learnable_downsample(x)
        outs.append(x)
        final_outs = []
        for i, out in enumerate(outs):
            if i in self.out_indices:
                out = out.permute(0, 2, 3, 1).contiguous()
                norm_layer = getattr(self, f'out_norm{i}')
                out = norm_layer(out)
                final_outs.append(out.permute(0, 3, 1, 2).contiguous())
        del outs
        return final_outs

    def forward(self, x):
        """Forward functions"""
        outs = self.forward_features(x)
        return outs

    def get_last_selfattention(self, x):
        """Returns last self-attention"""
        B = x.shape[0]
        # x is (B, N, C). (Hp, Hw) is (height in units of patches, width in units of patches)
        x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos_embed:
            # `pos_embed` (B, C, Hp, Wp), reshape -> (B, C, N), permute -> (B, N, C)
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, Hp, Wp)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        attn = None
        for i, blk in enumerate(self.cls_attn_blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                attn = blk(x, return_attention=True)
                return attn
        return attn


def fan_tiny_8_p4_hybrid(out_indices=[1, 2, 3], activation_checkpoint=True, **kwargs):
    """FAN Hybrid Tiny

    Args:
        out_indices (list): List of block indices to return as feature
    """
    depth = 8
    model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
    model = FAN(patch_size=16, in_chans=3, num_classes=80, embed_dim=192, depth=depth, backbone=backbone,
                num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3,
                act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=True,
                out_index=7, out_channels=[128, 256, 192, 768], out_indices=out_indices,
                activation_checkpoint=activation_checkpoint, **kwargs)

    return model


def fan_small_12_p4_hybrid(out_indices=[1, 2, 3], activation_checkpoint=True, **kwargs):
    """FAN Hybrid Small

    Args:
        out_indices (list): List of block indices to return as feature
    """
    depth = 10
    model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
    model = FAN(patch_size=16, in_chans=3, num_classes=80, embed_dim=384, depth=depth, backbone=backbone,
                num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3,
                act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=True,
                out_index=9, out_channels=[128, 256, 384, 768], out_indices=out_indices,
                activation_checkpoint=activation_checkpoint, **kwargs)

    return model


def fan_base_12_p4_hybrid(out_indices=[1, 2, 3], activation_checkpoint=True, **kwargs):
    """FAN Hybrid Base

    Args:
        out_indices (list): List of block indices to return as feature
    """
    depth = 16
    model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
    model = FAN(patch_size=16, in_chans=3, num_classes=80, embed_dim=448, depth=depth, backbone=backbone,
                num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3,
                act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=True,
                out_index=15, out_channels=[128, 256, 448, 768], out_indices=out_indices,
                activation_checkpoint=activation_checkpoint, **kwargs)

    return model


def fan_large_12_p4_hybrid(out_indices=[1, 2, 3], activation_checkpoint=True, **kwargs):
    """FAN Hybrid Large

    Args:
        out_indices (list): List of block indices to return as feature
    """
    depth = 22
    model_args = dict(depths=[3, 5], dims=[128, 256, 512, 1024], use_head=False)
    backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
    model = FAN(patch_size=16, in_chans=3, num_classes=80, embed_dim=480, depth=depth, backbone=backbone,
                num_heads=10, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3,
                act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=True,
                out_index=18, out_channels=[128, 256, 480, 768], out_indices=out_indices,
                activation_checkpoint=activation_checkpoint, **kwargs)

    return model


fan_model_dict = {
    'fan_tiny': fan_tiny_8_p4_hybrid,
    'fan_small': fan_small_12_p4_hybrid,
    'fan_base': fan_base_12_p4_hybrid,
    'fan_large': fan_large_12_p4_hybrid
}
