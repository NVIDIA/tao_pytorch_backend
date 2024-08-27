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

""" open_clip Model Module """

import math
import open_clip as OpenCLIP
import torch
from torch import nn
from open_clip import timm_model
from open_clip.transformer import VisionTransformer as OpenClip_VisionTransformer
from mmpretrain.registry import MODELS


@MODELS.register_module()
class open_clip(nn.Module):
    """
    open_clip model
    """

    def __init__(self, model_name="ViT-B-32", freeze=False, model_cfg=None, init_cfg=None, **kwargs):
        """
        Constructor for open_clip model
        """
        super().__init__()

        # Handle customized clip config
        if model_cfg is not None:
            OpenCLIP.factory._MODEL_CONFIGS[model_name] = model_cfg

        self.model = OpenCLIP.create_model(model_name, **kwargs)

        # To activate dynamic image size even after the model is created (initialized)
        if isinstance(self.model.visual, timm_model.TimmModel):
            # Override the attributes at instance level
            # These attributes are conditional atrributes that force timm's VisionTransformer to do interpolation of positional encoding inside the forward method
            self.model.visual.trunk.dynamic_img_size = True
            self.model.visual.trunk.patch_embed.strict_img_size = False
            self.model.visual.trunk.patch_embed.output_fmt = 'NHWC'
            self.model.visual.trunk.patch_embed.flatten = False
        elif isinstance(self.model.visual, OpenClip_VisionTransformer):
            # Override the method at instance level
            # Since open_clip's VisionTransformer does not has its own interpolatinon of positional encoding, we need to create by ourself and override the forward method
            self.model.visual.forward = self.interpolated_forward.__get__(self.model.visual, OpenClip_VisionTransformer)

        self.freeze = freeze

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def __interpolate_pos_encoding(self, w, h):
        """
        Interpolate the position encodings for the model to accept any image size.
        """
        npatch = (w * h) // (self.model.visual.patch_size[0] ** self.model.visual.patch_size[1])
        N = self.model.visual.positional_embedding.shape[0] - 1

        if npatch == N and w == h:
            return self.model.visual.positional_embedding

        class_pos_embed = self.model.visual.positional_embedding[0, :]
        patch_pos_embed = self.model.visual.positional_embedding[1:, :]
        dim = self.model.visual.positional_embedding.shape[-1]
        w0 = w // self.model.visual.patch_size[0]
        h0 = h // self.model.visual.patch_size[1]
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
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=0)

    # The function is modified from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py#L502 in order to support dynamic input size
    def interpolated_forward(self, x):
        """
        Forward using interpolated positional encodings
        """
        _, _, W, H = x.shape
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        expand_token = self.model.visual.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([expand_token.to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        dynamic_positional_embedding = self.__interpolate_pos_encoding(int(W), int(H))  # Support dynamic input size by interpolating the positional encoding
        x = x + dynamic_positional_embedding.to(x.dtype)

        x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        if self.model.visual.attn_pool is not None:
            if self.model.visual.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.model.visual.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.model.visual.attn_pool(x)
                if self.model.visual.attn_pool_type == 'parallel':
                    pooled = self.model.visual.attn_pool_contrastive(x)
                else:
                    assert self.model.visual.attn_pool_type == 'cascade'
                    pooled = self.model.visual.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.model.visual.attn_pool(x)
                x = self.model.visual.ln_post(x)
                pooled, tokens = self.model.visual._global_pool(x)
        elif self.model.visual.final_ln_after_pool:
            pooled, tokens = self.model.visual._global_pool(x)
            pooled = self.model.visual.ln_post(pooled)
        else:
            x = self.model.visual.ln_post(x)
            pooled, tokens = self.model.visual._global_pool(x)

        if self.model.visual.proj is not None:
            pooled = pooled @ self.model.visual.proj

        if self.model.visual.output_tokens:
            return pooled, tokens

        return pooled

    def forward(self, x):
        """
        Forward function and return the features
        """
        if self.freeze:
            self.eval()

        return self.model.encode_image(x, normalize=False)
