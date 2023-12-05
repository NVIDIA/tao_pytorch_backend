# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE


"""FAN Transformer Backbone Module for Segmentation."""

import math
from functools import partial
import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, trunc_normal_, to_2tuple

from nvidia_tao_pytorch.cv.ocdnet.model.backbone.convnext_utils import _create_hybrid_backbone
from nvidia_tao_pytorch.cv.backbone.fan import (PositionalEncodingFourier, Mlp, ConvPatchEmbed,
                                                ClassAttentionBlock, adaptive_avg_pool)
from nvidia_tao_pytorch.cv.ocdnet.utils.linear_activation import LinearActivation

QUANT = True
if QUANT:
    from pytorch_quantization.nn import QuantLinear, TensorQuantizer


class TokenMixing(nn.Module):
    """Token Mixing"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., quant=False, fuse_qkv_proj=True):
        """Init Function"""
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.do_quant = quant
        self.fuse_qkv_proj = fuse_qkv_proj
        # config of mlp for v processing
        if QUANT and self.do_quant:
            self.q = LinearActivation(dim, dim, bias=qkv_bias)
            self.kv = LinearActivation(dim, dim * 2, bias=qkv_bias)
            self.proj = LinearActivation(dim, dim)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            if self.fuse_qkv_proj:
                self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            else:
                self.k = nn.Linear(dim, dim, bias=qkv_bias)
                self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

        if QUANT and self.do_quant:
            self.matmul_q_input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
            self.matmul_k_input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
            self.matmul_v_input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
            self.matmul_a_input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
            self.softmax_input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)

    def _init_weights(self, m):
        """Init Weights"""
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

    def forward(self, x, H, W):
        """Forward Function"""
        B, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.fuse_qkv_proj:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if QUANT and self.do_quant:
            q_quant = self.matmul_q_input_quantizer(q * self.scale)
            k_quant = self.matmul_k_input_quantizer(k).transpose(-2, -1)
            attn = q_quant @ k_quant
            attn = self.softmax_input_quantizer(attn)
        else:
            attn = (q * self.scale @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        Bv, Hd, Nv, Cv = v.shape # noqa pylint: disable=W0612
        if QUANT and self.do_quant:
            x = (self.matmul_a_input_quantizer(attn) @ self.matmul_v_input_quantizer(v)).transpose(1, 2).reshape(B, N, self.dim)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, patch_size=2, feature_size=None, in_chans=3, embed_dim=384):
        """Init Function"""
        super().__init__()
        assert isinstance(backbone, nn.Module), "Backbone is not of instance Module."
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
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0, "Feature size is not a multiple of patch size."
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, return_feat=False):
        """Forward Function"""
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
                 drop_path=0., mlp_hidden_dim=None, act_layer=nn.GELU, drop=None, norm_layer=nn.LayerNorm):
        """Initialize ChannelProcessing class"""
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_v = norm_layer(dim)

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

    def _gen_attn(self, q, k, mode='none', shift_range=4, sampling_step=4):
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

        attn = torch.nn.functional.sigmoid(q @ k)
        return attn * self.temperature

    def forward(self, x, H, W, atten=None):
        """Forward functions """
        B, N, _ = x.shape
        v = x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads,  self.head_dim).permute(0, 2, 1, 3)

        attn = self._gen_attn(q, k)
        attn = self.attn_drop(attn)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd * Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)
        repeat_time = N // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, self.dim)
        return x,  attn * v.transpose(-1, -2)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Ignore during weight decay"""
        return {'temperature'}


class FANBlock(nn.Module):
    """FAN block from https://arxiv.org/abs/2204.12451"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, eta=1., quant=False, fuse_qkv_proj=True):
        """Initialize FANBlock class"""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenMixing(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, quant=quant, fuse_qkv_proj=fuse_qkv_proj)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.do_quant = quant
        mlp_block = ChannelProcessing
        self.mlp = mlp_block(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                             drop_path=drop_path, drop=drop, mlp_hidden_dim=int(dim * mlp_ratio))
        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

        self.H = None
        self.W = None
        if QUANT and self.do_quant:
            self.layernorm_input1_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
            self.add1_local_input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
            self.add1_residual_input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
            self.layernorm_input2_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
            self.add2_local_input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
            self.add2_residual_input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)

    def forward(self, x, attn=None, return_attention=False):
        """Forward Function"""
        H, W = self.H, self.W
        if QUANT and self.do_quant:
            x_new, attn = self.attn(self.norm1(self.layernorm_input1_quantizer(x)), H, W)
        else:
            x_new, attn = self.attn(self.norm1(x), H, W)
        if QUANT and self.do_quant:
            x = self.add1_residual_input_quantizer(x) + self.add1_local_input_quantizer(self.drop_path(self.gamma1 * x_new))
        else:
            x = x + self.drop_path(self.gamma1 * x_new)
        if QUANT and self.do_quant:
            x_new, attn = self.mlp(self.norm2(self.layernorm_input2_quantizer(x)), H, W, atten=attn)
        else:
            x_new, attn = self.mlp(self.norm2(x), H, W, atten=attn)
        if QUANT and self.do_quant:
            x = self.add2_residual_input_quantizer(x) + self.add2_local_input_quantizer(self.drop_path(self.gamma2 * x_new))
        else:
            x = x + self.drop_path(self.gamma2 * x_new)
        self.H, self.W = H, W
        if return_attention:
            return attn
        return x


class FAN(nn.Module):
    """Based on timm https://github.com/rwightman/pytorch-image-models/tree/master/timm"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., backbone=None, out_idx=-1,
                 act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=False, feat_downsample=False, use_checkpoint=False,
                 default_cfg=None, quant=False, fuse_qkv_proj=True, **kwargs):
        """Init Function"""
        super().__init__()
        img_size = to_2tuple(img_size)
        self.feat_downsample = feat_downsample
        self.use_checkpoint = use_checkpoint
        self.default_cfg = default_cfg
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.out_idx = out_idx
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        if backbone is None:
            self.patch_embed = ConvPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, act_layer=act_layer)
        else:
            self.patch_embed = HybridEmbed(backbone=backbone, patch_size=2, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        build_block = FANBlock
        self.blocks = nn.ModuleList([
            build_block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer, eta=eta, quant=quant, fuse_qkv_proj=fuse_qkv_proj)
            for i in range(depth)])

        self.cls_attn_blocks = nn.ModuleList([
            ClassAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, act_layer=act_layer, norm_layer=norm_layer, eta=eta, tokens_norm=tokens_norm)
            for _ in range(cls_attn_layers)])

        if isinstance(self.patch_embed, HybridEmbed) and feat_downsample:
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
        """Init Weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Ignore jit compile"""
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        """Function to get classifier head."""
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        """Resets head of classifier with num_classes"""
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """Extract Features."""
        outs = []
        out_index = [4, 7, 11]
        B = x.shape[0]
        # x is (B, N, C). (Hp, Hw) is (height in units of patches, width in units of patches)
        if isinstance(self.patch_embed, HybridEmbed):
            x, (Hp, Wp), out_list = self.patch_embed(x, return_feat=True)
            outs = outs + out_list
            out_index = [self.out_idx]
        else:
            x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos_embed:
            # `pos_embed` (B, C, Hp, Wp), reshape -> (B, C, N), permute -> (B, N, C)
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            blk.H, blk.W = Hp, Wp
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            Hp, Wp = blk.H, blk.W
            if idx in out_index:
                outs.append(x.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous())

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.cls_attn_blocks:
            x = blk(x)
        tmp = x[:, 1:, :].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()
        if isinstance(self.patch_embed, HybridEmbed) and self.feat_downsample:
            tmp = self.learnable_downsample(tmp)
            outs.append(tmp)
        else:
            outs.append(x[:, 1:, :].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous())

        return outs

    def forward(self, x):
        """Forward Function"""
        x = self.forward_features(x)
        return x

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

        for i, blk in enumerate(self.cls_attn_blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
        return None


# FAN-Hybrid models
class fan_debug(FAN):
    """FAN Hybrid Tiny"""

    def __init__(self, enlarge_feature_map_size=False, quant=False, activation_checkpoint=False, **kwargs):
        """Init Function"""
        depth = 1
        embed_dim = 192
        patch_size = 2 if enlarge_feature_map_size else 4
        model_args = dict(depths=[1, 1], dims=[128, 256, 512, 1024], use_head=False, quant=quant, patch_size=patch_size)
        backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
        super(fan_debug, self).__init__(patch_size=16, in_chans=3, num_classes=1000, embed_dim=embed_dim, depth=depth, backbone=backbone, out_idx=0,
                                        num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                                        act_layer=None, norm_layer=None, cls_attn_layers=1, use_pos_embed=True, eta=1., tokens_norm=True,
                                        use_checkpoint=activation_checkpoint, quant=quant, **kwargs)
        self.out_channels = model_args['dims'][:len(model_args['depths'])] + [embed_dim, embed_dim]


class fan_tiny_8_p4_hybrid(FAN):
    """FAN Hybrid Tiny"""

    def __init__(self, enlarge_feature_map_size=False, quant=False, activation_checkpoint=False, **kwargs):
        """Init Function"""
        depth = 8
        embed_dim = 192
        patch_size = 2 if enlarge_feature_map_size else 4
        model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False, quant=quant, patch_size=patch_size)
        backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
        super(fan_tiny_8_p4_hybrid, self).__init__(patch_size=16, in_chans=3, num_classes=1000, embed_dim=embed_dim, depth=depth, backbone=backbone, out_idx=7,
                                                   num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                                                   act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=True, use_checkpoint=activation_checkpoint, quant=quant, **kwargs)
        self.out_channels = model_args['dims'][:len(model_args['depths'])] + [embed_dim, embed_dim]


class fan_small_12_p4_hybrid(FAN):
    """FAN Hybrid Small"""

    def __init__(self, enlarge_feature_map_size=False, quant=False, activation_checkpoint=False, **kwargs):
        """Init Function"""
        depth = 10
        embed_dim = 384
        patch_size = 2 if enlarge_feature_map_size else 4
        model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False, quant=quant, patch_size=patch_size)
        backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
        super(fan_small_12_p4_hybrid, self).__init__(patch_size=16, in_chans=3, num_classes=1000, embed_dim=embed_dim, depth=depth, backbone=backbone, out_idx=9,
                                                     num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                                                     act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=True, use_checkpoint=activation_checkpoint, quant=quant, **kwargs)
        self.out_channels = model_args['dims'][:len(model_args['depths'])] + [embed_dim, embed_dim]


class fan_base_16_p4_hybrid(FAN):
    """FAN Hybrid Base"""

    def __init__(self, enlarge_feature_map_size=False, quant=False, activation_checkpoint=False, **kwargs):
        """Init Function"""
        depth = 16
        embed_dim = 448
        patch_size = 2 if enlarge_feature_map_size else 4
        model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False, quant=quant, patch_size=patch_size)
        backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
        super(fan_base_16_p4_hybrid, self).__init__(patch_size=16, in_chans=3, num_classes=1000, embed_dim=embed_dim, depth=depth, backbone=backbone, out_idx=15,
                                                    num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                                                    act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=True, use_checkpoint=activation_checkpoint, quant=quant, **kwargs)
        self.out_channels = model_args['dims'][:len(model_args['depths'])] + [embed_dim, embed_dim]


class fan_large_16_p4_hybrid(FAN):
    """FAN Hybrid Large"""

    def __init__(self, enlarge_feature_map_size=False, quant=False, activation_checkpoint=False, **kwargs):
        """Init Function"""
        depth = 22
        embed_dim = 480
        patch_size = 2 if enlarge_feature_map_size else 4
        model_args = dict(depths=[3, 5], dims=[128, 256, 512, 1024], use_head=False, quant=quant, patch_size=patch_size)
        backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
        super(fan_large_16_p4_hybrid, self).__init__(patch_size=16, in_chans=3, num_classes=1000, embed_dim=embed_dim, depth=depth, backbone=backbone, out_idx=18,
                                                     num_heads=10, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                                                     act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=True, use_checkpoint=activation_checkpoint, quant=quant, **kwargs)
        self.out_channels = model_args['dims'][:len(model_args['depths'])] + [embed_dim, embed_dim]
