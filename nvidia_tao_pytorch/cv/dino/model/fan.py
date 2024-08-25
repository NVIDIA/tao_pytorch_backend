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

"""FAN Backbone for DINO"""

import math
from functools import partial
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.layers import DropPath, trunc_normal_, to_2tuple

from nvidia_tao_pytorch.cv.backbone.convnext_utils import _create_hybrid_backbone
from nvidia_tao_pytorch.cv.backbone.fan import (PositionalEncodingFourier, Mlp, ConvPatchEmbed,
                                                ClassAttentionBlock, adaptive_avg_pool)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, patch_size=2, feature_size=None, in_chans=3, embed_dim=384):
        """Initialize HybridEmbedding class"""
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone.forward_features(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info[-1]['num_chs']
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, return_feat=False):
        """Forward function"""
        x, out_list = self.backbone.forward_features(x, return_feat=return_feat)
        _, _, H, W = x.shape
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        if return_feat:
            return x, (H // self.patch_size[0], W // self.patch_size[1]), out_list

        return x, (H // self.patch_size[0], W // self.patch_size[1])


class ChannelProcessing(nn.Module):
    """Channel Processing"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,
                 mlp_hidden_dim=None, act_layer=nn.GELU, drop=None, norm_layer=nn.LayerNorm, cha_sr_ratio=1):
        """Initialize ChannelProcessing class"""
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.cha_sr_ratio = cha_sr_ratio if num_heads > 1 else 1

        self.mlp_v = Mlp(in_features=dim // self.cha_sr_ratio, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_v = norm_layer(dim // self.cha_sr_ratio)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _gen_attn(self, q, k):
        """Returns attention"""
        _, _, N, _ = k.shape
        if torch.onnx.is_in_onnx_export():
            # If softmax dim is not the last dimension, then PyTorch decompose the softmax ops into
            # smaller ops like ReduceMax, ReduceSum, Sub, and Div.
            # As a result, ONNX export fails for opset_version >= 12.
            # Here, we rearrange the transpose so that softmax is done over the last dimension.
            q = q.transpose(-1, -2).softmax(-1)
            k = k.transpose(-1, -2).softmax(-1)
            warnings.warn("Replacing default adatpive_avg_pool2d to custom implementation for ONNX export")
            # adaptive_avg_pool2d is not supported for torch to onnx export
            k = adaptive_avg_pool(k.transpose(-1, -2), (N, 1))
        else:
            q = q.softmax(-2).transpose(-1, -2)
            k = torch.nn.functional.adaptive_avg_pool2d(k.softmax(-2), (N, 1))

        attn = torch.sigmoid(q @ k)
        return attn * self.temperature

    def forward(self, x, H, W):
        """Forward functions """
        _, N, C = x.shape
        v = x.reshape(-1, N, self.num_heads, C // self.num_heads // self.cha_sr_ratio).permute(0, 2, 1, 3)
        q = self.q(x).reshape(-1, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(-1, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

        attn = self._gen_attn(q, k)
        attn = self.attn_drop(attn)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd * Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)

        repeat_time = N // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(-1, N, C)

        return x,  attn * v.transpose(-1, -2)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Ignore during weight decay"""
        return {'temperature'}


class TokenMixing(nn.Module):
    """Token Mixing"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Initialize TokenMixing class"""
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # pylint:disable=I1101

        cha_sr = 1
        self.q = nn.Linear(dim, dim // cha_sr, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2 // cha_sr, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """Forward function"""
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        if torch.onnx.is_in_onnx_export() or not self.fast_attn:
            attn = (q * self.scale @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            # Since Torch 1.14, scaled_dot_product_attention has been optimized for performance
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FANBlock(nn.Module):
    """FAN block from https://arxiv.org/abs/2204.12451"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, eta=1.):
        """Initialize FANBlock class"""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenMixing(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = ChannelProcessing(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                     drop=drop, mlp_hidden_dim=int(dim * mlp_ratio))

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

        self.H = None
        self.W = None

    def forward(self, x, attn=None, return_attention=False):
        """Forward function"""
        H, W = self.H, self.W
        x_new = self.attn(self.norm1(x))
        x = x + self.drop_path(self.gamma1 * x_new)
        x_new, attn = self.mlp(self.norm2(x), H, W)
        x = x + self.drop_path(self.gamma2 * x_new)
        self.H, self.W = H, W
        if return_attention:
            return attn
        return x


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
                x = checkpoint.checkpoint(blk, x, use_reentrant=True)

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
        final_outs = {}
        for i, out in enumerate(outs):
            if i in self.out_indices:
                out = out.permute(0, 2, 3, 1).contiguous()
                norm_layer = getattr(self, f'out_norm{i}')
                out = norm_layer(out)
                final_outs[f'p{i}'] = out.permute(0, 3, 1, 2).contiguous()
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


def checkpoint_filter_fn(state_dict, model):
    """Filter loaded checkpoints"""
    if 'model' in state_dict:
        state_dict = state_dict['model']
    use_pos_embed = getattr(model, 'pos_embed', None) is not None
    pos_embed_keys = [k for k in state_dict if k.startswith('pos_embed')]
    for k in pos_embed_keys:
        if use_pos_embed:
            state_dict[k.replace('pos_embeder.', 'pos_embed.')] = state_dict.pop(k)
        else:
            del state_dict[k]
    if 'cls_attn_blocks.0.attn.qkv.weight' in state_dict and 'cls_attn_blocks.0.attn.q.weight' in model.state_dict():
        num_ca_blocks = len(model.cls_attn_blocks)
        for i in range(num_ca_blocks):
            qkv_weight = state_dict.pop(f'cls_attn_blocks.{i}.attn.qkv.weight')
            qkv_weight = qkv_weight.reshape(3, -1, qkv_weight.shape[-1])
            for j, subscript in enumerate('qkv'):
                state_dict[f'cls_attn_blocks.{i}.attn.{subscript}.weight'] = qkv_weight[j]
            qkv_bias = state_dict.pop(f'cls_attn_blocks.{i}.attn.qkv.bias', None)
            if qkv_bias is not None:
                qkv_bias = qkv_bias.reshape(3, -1)
                for j, subscript in enumerate('qkv'):
                    state_dict[f'cls_attn_blocks.{i}.attn.{subscript}.bias'] = qkv_bias[j]
    return state_dict


def fan_tiny_8_p4_hybrid(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
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


def fan_small_12_p4_hybrid(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
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


def fan_base_12_p4_hybrid(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
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


def fan_large_12_p4_hybrid(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
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
