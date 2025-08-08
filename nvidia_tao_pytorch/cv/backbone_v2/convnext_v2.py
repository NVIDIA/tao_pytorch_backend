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

"""ConvNextV2 backbone."""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase
from nvidia_tao_pytorch.cv.backbone_v2.nn.norm import GlobalResponseNorm, LayerNorm2d


class Block(nn.Module):
    """ConvNeXt and ConvNeXtV2 block.

    If `use_grn=True`, use `GlobalResponseNorm` instead of layer scale which is the ConvNeXtV2 style.
    """

    def __init__(self, dim, drop_path=0.0, use_grn=True, layer_scale_init_value=0.0):
        """Initialize the ConvNeXtV2 block.

        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic depth rate. Default: `0.0`.
            use_grn (bool): Whether to use `GlobalResponseNorm`. Default: `True`.
            layer_scale_init_value (float): Init value for Layer Scale. Default: `0.0`.
        """
        super().__init__()
        self.use_grn = bool(use_grn)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        if use_grn:
            self.grn = GlobalResponseNorm(4 * dim)
            self.gamma = None
        else:
            self.grn = nn.Identity()
            self.gamma = (
                nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
                if layer_scale_init_value > 0
                else None
            )
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Forward."""
        input_tensor = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.use_grn:
            x = self.grn(x)
        x = self.pwconv2(x)
        if not self.use_grn and self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input_tensor + self.drop_path(x)
        return x


class ConvNeXtV2(BackboneBase):
    """ConvNeXt V1 and V2 model.

    ConvNeXts is ConvNets that modernizes the classic ResNet design by incorporating elements inspired by vision
    Transformers. This results in a model that achieves competitive accuracy and scalability compared to Transformers,
    while retaining the simplicity and efficiency of ConvNets.

    ConvNeXt V2 introduces the Global Response Normalization (GRN) layer to the ConvNeXt architecture to enhance
    inter-channel feature competition.

    Reference:
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
    - [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)
    - [https://github.com/facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
    - [https://github.com/facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        # `use_grn` and `layer_scale_init_value` are for switching between ConvNeXt and ConvNeXtV2.
        # ConvNeXt uses `use_grn=False` and `layer_scale_init_value=1e-6`.
        # ConvNeXtV2 uses `use_grn=True` and `layer_scale_init_value=0.0`.
        use_grn=True,
        layer_scale_init_value=0.0,
        drop_path_rate=0.0,
        head_init_scale=1.0,
        export_pre_logits=False,
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        **kwargs,
    ):
        """Initialize the ConvNextV2 model.

        Args:
            in_chans (int): Number of input image channels. Default: `3`.
            num_classes (int): Number of classes for classification head. Default: `1000`.
            depths (tuple(int)): Number of blocks at each stage. Default: `[3, 3, 9, 3]`.
            dims (tuple(int)): Feature dimension at each stage. Default: `[96, 192, 384, 768]`.
            use_grn (bool): Whether to use `GlobalResponseNorm`. Default: `True`.
            layer_scale_init_value (float): Init value for Layer Scale. Default: `0.0`.
            drop_path_rate (float): Stochastic depth rate. Default: `0`.
            head_init_scale (float): Init scaling value for classifier weights and biases. Default: `1`.
            export_pre_logits (bool): Whether to export the pre_logits features of the model. Default: `False`.
            activation_checkpoint (bool): Whether to use activation checkpointing. Default: `False`.
            freeze_at (list): List of keys corresponding to the stages or layers to freeze. If `None`, no specific
                layers are frozen. If `"all"`, the entire model is frozen and set to eval mode. Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the backbone will be frozen. Default: `False`.
        """
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )
        self.depths = depths
        self.dims = dims
        self.use_grn = use_grn
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate
        self.head_init_scale = head_init_scale
        self.export_pre_logits = export_pre_logits

        self.num_features = dims[-1]
        self.num_stages = len(depths)

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        use_grn=self.use_grn,
                        layer_scale_init_value=self.layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        if num_classes > 0:
            self.head = nn.Linear(dims[-1], num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Set the gradient checkpointing for the model."""
        self.activation_checkpoint = enable

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {0: self.downsample_layers[0]}
        for i, stage in enumerate(self.stages, start=1):
            # TODO(yuw, hongyuc): Should we freeze the downsample layers?
            stage_dict[i] = stage
        return stage_dict

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classifier module."""
        return self.head

    def reset_classifier(self, num_classes):
        """Reset the classifier head."""
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
            self.head.weight.data.mul_(self.head_init_scale)
            self.head.bias.data.mul_(self.head_init_scale)
        else:
            self.head = nn.Identity()

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head."""
        for idx in range(self.num_stages):
            x = self.downsample_layers[idx](x)
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = self.stages[idx](x)
            else:
                x = checkpoint.checkpoint(self.stages[idx], x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward_feature_pyramid(self, *args, **kwargs):
        """Forward pass through the backbone to extract intermediate feature maps."""
        raise NotImplementedError("forward_feature_pyramid is not implemented.")

    def forward(self, x):
        """Forward."""
        x = self.forward_pre_logits(x)
        if self.export_pre_logits:
            return x
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def convnextv2_atto(**kwargs):
    """Constructs a ConvNextV2-Atto model."""
    return ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_femto(**kwargs):
    """Constructs a ConvNextV2-Femto model."""
    return ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_pico(**kwargs):
    """Constructs a ConvNextV2-Pico model."""
    return ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_nano(**kwargs):
    """Constructs a ConvNextV2-Nano model."""
    return ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_tiny(**kwargs):
    """Constructs a ConvNextV2-Tiny model."""
    return ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_base(**kwargs):
    """Constructs a ConvNextV2-Base model."""
    return ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_large(**kwargs):
    """Constructs a ConvNextV2-Large model."""
    return ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_huge(**kwargs):
    """Constructs a ConvNextV2-Huge model."""
    return ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
