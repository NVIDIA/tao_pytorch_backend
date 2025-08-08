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

"""OpenCLIP backbone."""

import math
import types

import open_clip
import torch
import torch.nn as nn
from open_clip import timm_model
from open_clip.transformer import VisionTransformer as OpenCLIPVisionTransformer

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


NVCLIP_COMMON_VISION_CONFIG = {
    "patch_size": 14,
    "no_ln_pre": True,
    "pool_type": "avg",
    "final_ln_after_pool": True,
    "pos_embed_type": "sin_cos_2d",
    "patch_dropout": 0.0,
}
NVCLIP_COMMON_TEXT_CONFIG = {
    "context_length": 77,
    "vocab_size": 32000,
    "hf_tokenizer_name": "bert-base-uncased",
    "tokenizer_kwargs": {"strip_sep_token": True},
    "pool_type": "last",
    "no_causal_mask": True,
}
NVCLIP_MODEL_CONFIG = {
    "ViT-H-14-SigLIP-CLIPA-224": {
        "embed_dim": 1024,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 224,
            "layers": 32,
            "width": 1280,
            "head_width": 80,
            **NVCLIP_COMMON_VISION_CONFIG,
        },
        "text_cfg": {"width": 1024, "heads": 16, "layers": 24, **NVCLIP_COMMON_TEXT_CONFIG},
    },
    "ViT-L-14-SigLIP-CLIPA-336": {
        "embed_dim": 768,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 336,
            "layers": 24,
            "width": 1024,
            "head_width": 64,
            **NVCLIP_COMMON_VISION_CONFIG,
        },
        "text_cfg": {"width": 768, "heads": 12, "layers": 12, **NVCLIP_COMMON_TEXT_CONFIG},
    },
    "ViT-L-14-SigLIP-CLIPA-224": {
        "embed_dim": 768,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 224,
            "layers": 24,
            "width": 1024,
            "head_width": 64,
            **NVCLIP_COMMON_VISION_CONFIG,
        },
        "text_cfg": {"width": 768, "heads": 12, "layers": 12, **NVCLIP_COMMON_TEXT_CONFIG},
    },
}


class OpenCLIP(BackboneBase):
    """OpenCLIP model.

    This class is mostly for NV-CLIP models.

    NV-CLIP is a multimodal embeddings model for image and text. Trained on 700M proprietary images, NV-CLIP is the
    NVIDIA commercial version of OpenAI CLIP (Contrastive Language-Image Pre-Training) model. NV-CLIP can be applied
    to various areas such as multimodal search, zero-shot image classification, and downstream computer vision tasks
    such as object detection and more.

    References:
    - [https://build.nvidia.com/nvidia/nvclip/modelcard](https://build.nvidia.com/nvidia/nvclip/modelcard)
    - [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 0,
        model_name: str = "ViT-L-14-SigLIP-CLIPA-336",
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        **kwargs,
    ):
        """Initialize the OpenCLIP model.

        Args:
            in_chans (int): Number of input image channels. Default: `3`.
            num_classes (int): Number of classes for classification head. Default: `0`.
            model_name (str): The name of the model to load. Default: `"ViT-L-14-SigLIP-CLIPA-336"`.
            activation_checkpoint (bool): Whether to use activation checkpointing. Default: `False`.
            freeze_at (list): List of keys corresponding to the stages or layers to freeze. If `None`, no specific
                layers are frozen. Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the backbone will be frozen. Default: `False`.
        """
        if in_chans != 3:
            raise ValueError(f"in_chans must be 3 for OpenCLIP backbones. Received: in_chans={in_chans}")

        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )
        self._register_nvclip_configs()
        self.model = open_clip.create_model(model_name, **kwargs)

        # Enable dynamic image size after the model is initialized.
        if isinstance(self.model.visual, timm_model.TimmModel):
            self.model.visual.trunk.dynamic_img_size = True
            self.model.visual.trunk.patch_embed.strict_img_size = False
            self.model.visual.trunk.patch_embed.output_fmt = "NHWC"
            self.model.visual.trunk.patch_embed.flatten = False
        elif isinstance(self.model.visual, OpenCLIPVisionTransformer):
            self._enable_interpolated_forward(self.model.visual)
        else:
            raise NotImplementedError(f"Unsupported model type {model_name} for dynamic image size.")
        self.head = nn.Linear(self.model.visual.output_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _register_nvclip_configs(self):
        """Register NVCLIP model configurations."""
        for model_name, model_cfg in NVCLIP_MODEL_CONFIG.items():
            if model_name not in open_clip.factory._MODEL_CONFIGS:
                open_clip.factory._MODEL_CONFIGS[model_name] = model_cfg

    def _enable_interpolated_forward(self, model: OpenCLIPVisionTransformer):
        """Enable interpolated forward for the model."""

        def interpolate_pos_encoding(self: OpenCLIPVisionTransformer, w, h):
            """Interpolate the position encodings for the model to accept any image size."""
            npatch = (w * h) // (self.patch_size[0] * self.patch_size[1])
            N = self.positional_embedding.shape[0] - 1
            if npatch == N and w == h:
                return self.positional_embedding
            class_pos_embed = self.positional_embedding[0, :]
            patch_pos_embed = self.positional_embedding[1:, :]
            dim = self.positional_embedding.shape[-1]
            w0 = w // self.patch_size[0]
            h0 = h // self.patch_size[1]
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            # We need fp32 for the interpolation
            reshaped_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            )
            patch_pos_embed = nn.functional.interpolate(
                reshaped_pos_embed,
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1], (
                "The interpolated value does not match the positional embedding size."
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=0)

        def interpolated_forward(self: OpenCLIPVisionTransformer, x: torch.Tensor):
            """Forward using interpolated positional encodings."""
            _, _, W, H = x.shape
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

            # class embeddings and positional embeddings
            expand_token = self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1)
            x = torch.cat([expand_token.to(x.dtype), x], dim=1)
            # shape = [*, grid ** 2 + 1, width]
            # Support dynamic input size by interpolating the positional encoding
            dynamic_positional_embedding = interpolate_pos_encoding(self, int(W), int(H))
            x = x + dynamic_positional_embedding.to(x.dtype)

            x = self.patch_dropout(x)
            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            if self.attn_pool is not None:
                if self.attn_pool_contrastive is not None:
                    # This is untested, WIP pooling that should match paper
                    x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                    tokens = self.attn_pool(x)
                    if self.attn_pool_type == "parallel":
                        pooled = self.attn_pool_contrastive(x)
                    else:
                        assert self.attn_pool_type == "cascade"
                        pooled = self.attn_pool_contrastive(tokens)
                else:
                    # this is the original OpenCLIP CoCa setup, does not match paper
                    x = self.attn_pool(x)
                    x = self.ln_post(x)
                    pooled, tokens = self._global_pool(x)
            elif self.final_ln_after_pool:
                pooled, tokens = self._global_pool(x)
                pooled = self.ln_post(pooled)
            else:
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
            if self.proj is not None:
                pooled = pooled @ self.proj
            if self.output_tokens:
                return pooled, tokens
            return pooled

        model.forward = types.MethodType(interpolated_forward, model)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Set the gradient checkpointing for the model."""
        self.model.set_grad_checkpointing(enable)

    def load_state_dict(self, state_dict, **kwargs):
        """Copy parameters and buffers from state_dict into this module and its descendants.

        Args:
            state_dict (dict): a dict containing parameters and persistent buffers.
            **kwargs: Additional arguments passed to `nn.Module.load_state_dict`.
        """
        return self.model.load_state_dict(state_dict, **kwargs)

    def get_stage_dict(self):
        """Get the stage dictionary."""
        if isinstance(self.model.visual, timm_model.TimmModel):
            # TODO(@hongyuc): Does TimmModel have a unified signature?
            raise NotImplementedError("Stage dictionary is not implemented for TimmModel.")
        elif isinstance(self.model.visual, OpenCLIPVisionTransformer):
            stage_dict = {0: self.model.visual.conv1}
            for i, block in enumerate(self.model.visual.transformer.resblocks, start=1):
                stage_dict[i] = block
        else:
            raise NotImplementedError("Stage dictionary is not implemented for this model type.")
        return stage_dict

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classification head module.

        Returns:
            nn.Module: The classification head (Linear layer or Identity).
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        """Reset the classification head with a new number of classes.

        Args:
            num_classes (int): New number of classes for classification.
            global_pool (str, optional): Global pooling type (unused in current implementation).
                Defaults to "".
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_pre_logits(self, x):
        """Forward pass through the visual encoder.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Features of shape (B, D).
        """
        return self.model.encode_image(x, normalize=False)

    def forward_feature_pyramid(self, *args, **kwargs):
        """Forward pass through the backbone to extract intermediate feature maps."""
        raise NotImplementedError("forward_feature_pyramid is not implemented.")

    def forward(self, x):
        """Forward pass through the visual encoder.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Features of shape (B, D).
        """
        x = self.model.encode_image(x, normalize=False)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def vit_l_14_siglip_clipa_224(**kwargs):
    """ViT Large Patch14 SigLIP CLIPA 224 model."""
    return OpenCLIP(model_name="ViT-L-14-SigLIP-CLIPA-224", **kwargs)


@BACKBONE_REGISTRY.register()
def vit_l_14_siglip_clipa_336(**kwargs):
    """ViT Large Patch14 SigLIP CLIPA 336 model."""
    return OpenCLIP(model_name="ViT-L-14-SigLIP-CLIPA-336", **kwargs)


@BACKBONE_REGISTRY.register()
def vit_h_14_siglip_clipa_224(**kwargs):
    """ViT Huge Patch14 SigLIP CLIPA 224 model."""
    return OpenCLIP(model_name="ViT-H-14-SigLIP-CLIPA-224", **kwargs)
