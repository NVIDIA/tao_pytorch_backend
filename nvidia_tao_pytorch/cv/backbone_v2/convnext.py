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

"""ConvNeXt backbone."""

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.convnext_v2 import ConvNeXtV2 as ConvNeXt


@BACKBONE_REGISTRY.register()
def convnext_tiny(**kwargs):
    """Constructs a ConvNext-Tiny model."""
    return ConvNeXt(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], use_grn=False, layer_scale_init_value=1e-6, **kwargs
    )


@BACKBONE_REGISTRY.register()
def convnext_small(**kwargs):
    """Constructs a ConvNext-Small model."""
    return ConvNeXt(
        depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], use_grn=False, layer_scale_init_value=1e-6, **kwargs
    )


@BACKBONE_REGISTRY.register()
def convnext_base(**kwargs):
    """Constructs a ConvNext-Base model."""
    return ConvNeXt(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], use_grn=False, layer_scale_init_value=1e-6, **kwargs
    )


@BACKBONE_REGISTRY.register()
def convnext_large(**kwargs):
    """Constructs a ConvNext-Large model."""
    return ConvNeXt(
        depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], use_grn=False, layer_scale_init_value=1e-6, **kwargs
    )


@BACKBONE_REGISTRY.register()
def convnext_xlarge(**kwargs):
    """Constructs a ConvNext-XLarge model."""
    return ConvNeXt(
        depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], use_grn=False, layer_scale_init_value=1e-6, **kwargs
    )
