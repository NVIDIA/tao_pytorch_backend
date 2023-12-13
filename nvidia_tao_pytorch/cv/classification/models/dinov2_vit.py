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

""" DINOv2 ViT Model Module """

import math
from functools import partial

import torch
from torch import nn
from mmcls.models.builder import BACKBONES
from timm.layers import PatchEmbed, SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer


def interpolate_pos_encoding(pos_embed, w, h, patch_size=14):
    """
    Interpolate the position encodings for the model to accept any image size.
    """
    w0 = w // patch_size
    h0 = h // patch_size
    npatch = w0 * h0
    N = pos_embed.shape[1] - 1

    if npatch == N and w == h:
        return pos_embed

    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8

    w0, h0 = w0 + 0.1, h0 + 0.1

    # We need fp32 for the interpolation
    reshaped_pos_embed = patch_pos_embed.reshape(
        1, int(math.sqrt(N)), int(math.sqrt(N)), pos_embed.shape[-1]
    ).permute(0, 3, 1, 2)

    patch_pos_embed = nn.functional.interpolate(
        reshaped_pos_embed,
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode="bicubic",
    )

    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1], "The interpolated value does not match the positional embedding size."
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(
        1, -1, pos_embed.shape[-1]
    )

    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


class DinoV2ViT(VisionTransformer):
    """
    This class extends the VisionTransformer class from timm library so that we can
    handle different image sizes.
    """

    def __interpolate_pos_encoding(self, x, w, h):
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

        # We need fp32 for the interpolation
        reshaped_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(N)), int(math.sqrt(N)), dim
        ).permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            reshaped_pos_embed,
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1], "The interpolated value does not match the positional embedding size."
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def _pos_embed(self, x):
        B, S, _ = x.shape
        w = h = int(math.sqrt(S)) * self.patch_embed.patch_size[0]

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.__interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)


@BACKBONES.register_module()
class vit_large_patch14_dinov2_swiglu(DinoV2ViT):
    """
    DINOV2 ViT Large model with SwiGLU activation
    """

    def __init__(self, *args, freeze=False, pretrained=None, **kwargs):
        """Initialize"""
        model_kwargs = dict(
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
            num_classes=0,
            **kwargs
        )
        super(vit_large_patch14_dinov2_swiglu, self).__init__(*args, **model_kwargs)
        self.freeze = freeze

        if pretrained is not None:
            model = torch.load(pretrained, "cpu")
            self.load_state_dict(model, strict=True)
            print(f"Loaded pretrained weights from {pretrained}")

        if freeze:
            assert pretrained is not None, "You shouldn't freeze a model without specifying pretrained"
            self.eval()

            for p in self.parameters():
                p.requires_grad = False

    def forward(self, *args, **kwargs):
        """
        Forward function and return the flatten output (cls_token)
        """
        if self.freeze:
            self.eval()

        x = super().forward(*args, **kwargs)

        return x.flatten(1)
