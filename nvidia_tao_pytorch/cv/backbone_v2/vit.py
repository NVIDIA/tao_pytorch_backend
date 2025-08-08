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

"""ViT backbone."""

import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


class VisionTransformer(TimmVisionTransformer, BackboneBase):
    """Vision Transformer model.

    This class extends the VisionTransformer class from timm library so that we
    can handle different image sizes.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the ViT model.

        Args:
            img_size (list): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input channels.
            num_classes (int): Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim (int): Transformer embedding dimension.
            depth (int): Depth of transformer.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): Enable bias for qkv projections if True.
            init_values (float): Layer-scale init values (layer-scale enabled if not None).
            class_token (bool): Use class token.
            no_embed_class (bool): Don't include position embeddings for class (or reg) tokens.
            reg_tokens (int): Number of register tokens.
            pre_norm (bool): Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
            final_norm (bool): Enable norm after transformer blocks, before head (standard in most ViT).
            fc_norm (bool): Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate (float): Head dropout rate.
            pos_drop_rate (float): Position embedding dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float): Stochastic depth rate.
            weight_init (str): Weight initialization scheme.
            fix_init (bool): Apply weight initialization fix (scaling w/ layer index).
            embed_layer (Callable): Patch embedding layer.
            embed_norm_layer: Normalization layer to use / override in patch embed module.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
            activation_checkpoint (bool): Whether to use activation checkpointing. Default: `False`.
            freeze_at (list): List of keys corresponding to the stages or layers to freeze. If `None`, no specific
                layers are frozen. If `"all"`, the entire model is frozen and set to eval mode. Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the backbone will be frozen. Default: `False`.
        """
        in_chans = kwargs.get("in_chans", 3)
        num_classes = kwargs.get("num_classes", 1000)
        activation_checkpoint = kwargs.pop("activation_checkpoint", False)
        freeze_at = kwargs.pop("freeze_at", None)
        freeze_norm = kwargs.pop("freeze_norm", False)

        super().__init__(*args, **kwargs)  # TimmVisionTransformer initialization.
        BackboneBase.__init__(
            self,
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {0: self.patch_embed}
        for i, block in enumerate(self.blocks, start=1):
            stage_dict[i] = block
        return stage_dict

    def _interpolate_pos_encoding(self, x, w, h):
        """Interpolate Positional Encoding based on given resolution."""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        reshaped_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            reshaped_pos_embed,
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        embed_w = patch_pos_embed.shape[-2]
        embed_h = patch_pos_embed.shape[-1]
        if int(w0) != embed_w or int(h0) != embed_h:
            raise ValueError("The interpolated value does not match the positional embedding size.")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def get_spatial_feat(self, x):
        """Turn token feature into spatial feature.

        Args:
            x (torch.Tensor): token feature in [B, 1024+1, 768]
        Return:
            x (torch.Tensor): feature map in (B, 768, H, W)
        """
        b, n, c = x.shape
        h, w = int((n - 1 + 1e-6) ** 0.5), int((n - 1 + 1e-6) ** 0.5)
        x = x[:, 1:].transpose(2, 1).reshape(b, c, h, w)
        return x

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Features of shape (B, L, D).
        """
        x = super().forward_features(x)
        x = super().forward_head(x, pre_logits=True)
        return x

    def forward_feature_pyramid(self, *args, **kwargs):
        """Forward pass through the backbone to extract intermediate feature maps."""
        raise NotImplementedError("forward_feature_pyramid is not implemented.")

    def forward(self, x):
        """Forward.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


class VisionTransformerMAE(VisionTransformer):
    """Vision Transformer for MAE."""

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head."""
        x = super().forward_features(x)
        x = x[:, 1:, :].mean(dim=1)
        return x

    def forward(self, x: torch.Tensor):
        """Forward.

        This function returns the intermediate features of the model
        in backbone mode. The function is overriden from the base class.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C).
        """
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def vit_base_patch16(**kwargs):
    """ViT Base model."""
    return VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        final_norm=True,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_base_patch16_mae(**kwargs):
    """ViT Base model."""
    return VisionTransformerMAE(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        final_norm=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_large_patch16(**kwargs):
    """ViT Large model."""
    return VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        final_norm=True,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_large_patch16_mae(**kwargs):
    """ViT Large model."""
    return VisionTransformerMAE(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        final_norm=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_huge_patch14(**kwargs):
    """ViT Huge model."""
    return VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        final_norm=True,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_huge_patch14_mae(**kwargs):
    """ViT Huge model."""
    return VisionTransformerMAE(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        final_norm=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
