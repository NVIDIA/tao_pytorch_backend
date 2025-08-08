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

"""Hiera backbone."""

import torch
from timm.models.hiera import Hiera as TimmHiera

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


class Hiera(TimmHiera, BackboneBase):
    """Hiera model.

    Hiera is a hierarchical vision transformer that is fast, powerful, and, above all, simple. It outperforms the
    state-of-the-art across a wide array of image and video tasks while being much faster.

    References:
    - [Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](
      https://arxiv.org/abs/2306.00989)
    - [https://github.com/facebookresearch/hiera](https://github.com/facebookresearch/hiera)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Hiera model."""
        in_chans = kwargs.get("in_chans", 3)
        num_classes = kwargs.get("num_classes", 1000)
        activation_checkpoint = kwargs.pop("activation_checkpoint", False)
        freeze_at = kwargs.pop("freeze_at", None)
        freeze_norm = kwargs.pop("freeze_norm", False)

        super().__init__(*args, **kwargs)  # TimmHiera initialization.
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
        stage_dict = {}
        # TODO(@yuw, @hongyuc): No stem. Add patch_embed as stage 0?
        for i, block in enumerate(self.blocks, start=1):
            stage_dict[i] = block
        return stage_dict

    def forward_pre_logits(self, x: torch.Tensor, mask: torch.Tensor = None):
        """Forward pass through the backbone, excluding the head.

        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that
        dim.

        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """
        return super().forward_features(x, mask, return_intermediates=False)

    def forward_feature_pyramid(self, *args, **kwargs):
        """Forward pass through the backbone to extract intermediate feature maps."""
        raise NotImplementedError("forward_feature_pyramid is not implemented.")


@BACKBONE_REGISTRY.register()
def hiera_tiny_224(**kwargs):
    """Hiera tiny."""
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), **kwargs)


@BACKBONE_REGISTRY.register()
def hiera_small_224(**kwargs):
    """Hiera small."""
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2), **kwargs)


@BACKBONE_REGISTRY.register()
def hiera_base_224(**kwargs):
    """Hiera base."""
    return Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), **kwargs)


@BACKBONE_REGISTRY.register()
def hiera_base_plus_224(**kwargs):
    """Hiera base plus."""
    return Hiera(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwargs)


@BACKBONE_REGISTRY.register()
def hiera_large_224(**kwargs):
    """Hiera large."""
    return Hiera(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwargs)


@BACKBONE_REGISTRY.register()
def hiera_huge_224(**kwargs):
    """Hiera huge."""
    return Hiera(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwargs)
