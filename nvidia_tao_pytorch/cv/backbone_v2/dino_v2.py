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

"""DINOv2 backbone."""

import math
from functools import partial

import torch
import torch.nn as nn
from timm.layers import PatchEmbed, SwiGLUPacked

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.vit import VisionTransformer


class DINOV2(VisionTransformer):
    """DINOV2 model.

    This class extends the VisionTransformer by adding register tokens and
    handling different image sizes.

    References:
    - [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the DINOV2 model.

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
            register_tokens (int): Number of register tokens to be added.
            activation_checkpoint (bool): Whether to use activation checkpointing. Default: `False`.
            freeze_at (list): List of keys corresponding to the stages or layers to freeze. If `None`, no specific
                layers are frozen. If `"all"`, the entire model is frozen and set to eval mode. Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the backbone will be frozen. Default: `False`.
        """
        register_tokens = kwargs.pop("register_tokens", 0)

        super().__init__(*args, **kwargs)  # VisionTransformer initialization.

        # Add register tokens.
        self.num_register_tokens = register_tokens
        if register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, register_tokens, self.embed_dim))

    def _pos_embed(self, x):
        B, S, _ = x.shape
        w = h = int(math.sqrt(S)) * self.patch_embed.patch_size[0]
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional encoding to each token
        x = x + self._interpolate_pos_encoding(x, w, h)
        # add register tokens
        if self.num_register_tokens > 0:
            x = torch.cat((x, self.register_tokens.expand(B, -1, -1)), dim=1)
        return self.pos_drop(x)

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head."""
        x = super().forward_features(x)
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x.flatten(1)

    def forward(self, x):
        """Forward."""
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def vit_large_patch14_dinov2_swiglu(**kwargs):
    """DINOV2 ViT Large model with SwiGLU activation."""
    return DINOV2(
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-5,
        img_size=518,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        mlp_ratio=5472 / 1024,
        embed_layer=partial(PatchEmbed, strict_img_size=False),
        global_pool="token",
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_large_patch14_dinov2_swiglu_legacy(**kwargs):
    """DINOV2 ViT Large model with SwiGLU activation. Legacy version."""
    return DINOV2(
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-5,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        mlp_ratio=5472 / 1024,
        embed_layer=partial(PatchEmbed, strict_img_size=False),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_giant_patch14_reg4_dinov2_swiglu(**kwargs):
    """DINOV2 ViT Giant model with SwiGLU activation."""
    return DINOV2(
        patch_size=14,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        init_values=1e-5,
        img_size=518,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        mlp_ratio=8192 / 1536,
        embed_layer=partial(PatchEmbed, strict_img_size=False),
        global_pool="token",
        register_tokens=4,
        **kwargs,
    )
