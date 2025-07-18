# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""ConvNext backbone for RT-DETR. """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import trunc_normal_, DropPath

from nvidia_tao_pytorch.cv.deformable_detr.model.backbone import FrozenBatchNorm2d
from nvidia_tao_pytorch.ssl.mae.model.convnextv2 import ConvNeXtV2


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """Init function."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(
                f"Invalid data format: {self.data_format}"
                f"Valid options are ['channels_last', 'channels_first']"
            )
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        """Forward function."""
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        """Init function.

        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic depth rate. Default: 0.0
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """Forward function."""
        input_tensor = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_tensor + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
    A PyTorch impl of : `A ConvNet for the 2020s`  -
    https://arxiv.org/pdf/2201.03545.pdf
    """

    def __init__(self,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 num_stages=4,
                 freeze_at=0,
                 freeze_norm=True,
                 return_idx=[0, 1, 2, 3],
                 out_channels=[512, 1024, 2048],
                 in_chans=3,
                 drop_path_rate=0,
                 layer_scale_init_value=1e-6,
                 activation_checkpoint=False):
        """Init function.

        Args:
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
            dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
            drop_path_rate (float): Stochastic depth rate. Default: 0.
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
            head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        """
        super().__init__()

        self.num_stages = num_stages
        self.activation_checkpoint = activation_checkpoint
        # stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        for i in range(num_stages):
            if i == 0:
                stem = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                )
                self.downsample_layers.append(stem)
            else:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i - 1], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(num_stages):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                  layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        self.return_idx = return_idx
        self.out_channels = out_channels

        # TODO: @scha check if we can remove the hardcoded upsampling conv from Synthetica.
        # Add upsampling layer to match the dimension of Encoder input
        # self.conv_upsample = nn.ModuleList()
        # for i, out_c in enumerate(out_channels):
        #     conv = nn.Conv2d(
        #         dims[self.return_idx[i]],
        #         out_c,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1
        #     )
        #     self.conv_upsample.append(conv)

        assert len(self.return_idx) == 3, f"ConvNext only supports num_feature_levels == 3, Got {len(self.return_idx)}"
        self.conv_512 = nn.Conv2d(dims[self.return_idx[0]], 512, kernel_size=3, stride=1, padding=1)
        self.conv_1024 = nn.Conv2d(dims[self.return_idx[1]], 1024, kernel_size=3, stride=1, padding=1)
        self.conv_2048 = nn.Conv2d(dims[self.return_idx[2]], 2048, kernel_size=3, stride=1, padding=1)

        self.conv_upsample = []
        self.conv_upsample.append(self.conv_512)
        self.conv_upsample.append(self.conv_1024)
        self.conv_upsample.append(self.conv_2048)

        self.apply(self._init_weights)

        if freeze_at >= 0:
            self._freeze_parameters(self.downsample_layers[0])
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

    def _freeze_parameters(self, m: nn.Module):
        """freeze parameters."""
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        """freeze normalization."""
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _init_weights(self, m):
        """initialize weights."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward function."""
        outs = []

        for idx in range(self.num_stages):
            x = self.downsample_layers[idx](x)
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = self.stages[idx](x)
            else:
                x = checkpoint.checkpoint(self.stages[idx], x)

            if idx in self.return_idx:
                yUp = self.conv_upsample[idx - 1](x)
                outs.append(yUp)

        return outs


class ConvNeXtV2_FPN(ConvNeXtV2):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        activation_checkpoint (bool): Whether to use activation checkpointing. Default: False
        freeze_at (int): Number of stages to freeze. Default: 0
        freeze_norm (bool): Whether to freeze normalization. Default: True
        return_idx (list): List of block indices to return as feature. Default: [1, 2, 3]
        out_channels (list): List of output channels. Default: [512, 1024, 2048]
    """

    def __init__(self, freeze_at=0, freeze_norm=True,
                 return_idx=[1, 2, 3],
                 out_channels=[512, 1024, 2048],
                 activation_checkpoint=False, **kwargs):
        super().__init__(**kwargs)
        self.return_idx = return_idx
        self.out_channels = out_channels
        self.activation_checkpoint = activation_checkpoint
        assert len(self.return_idx) == 3, f"ConvNext only supports num_feature_levels == 3, Got {len(self.return_idx)}"
        self.conv_512 = nn.Conv2d(self.dims[self.return_idx[0]], 512, kernel_size=3, stride=1, padding=1)
        self.conv_1024 = nn.Conv2d(self.dims[self.return_idx[1]], 1024, kernel_size=3, stride=1, padding=1)
        self.conv_2048 = nn.Conv2d(self.dims[self.return_idx[2]], 2048, kernel_size=3, stride=1, padding=1)

        self.conv_upsample = []
        self.conv_upsample.append(self.conv_512)
        self.conv_upsample.append(self.conv_1024)
        self.conv_upsample.append(self.conv_2048)
        self.apply(self._init_weights)

        if freeze_at >= 0:
            self._freeze_parameters(self.downsample_layers[0])
            for i in range(min(freeze_at, self.num_stages)):
                self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

    def _freeze_parameters(self, m: nn.Module):
        """freeze parameters."""
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        """freeze normalization."""
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        """Forward function."""
        outs = []

        for idx in range(self.num_stages):
            x = self.downsample_layers[idx](x)
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = self.stages[idx](x)
            else:
                x = checkpoint.checkpoint(self.stages[idx], x)

            if idx in self.return_idx:
                yUp = self.conv_upsample[idx - 1](x)
                outs.append(yUp)

        return outs


def convnext_tiny(out_indices=[1, 2, 3], **kwargs):
    """ ConvNext-Tiny model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                    return_idx=out_indices, **kwargs)


def convnext_small(out_indices=[1, 2, 3], **kwargs):
    """ ConvNext-Small model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768],
                    return_idx=out_indices, **kwargs)


def convnext_base(out_indices=[1, 2, 3], **kwargs):
    """ ConvNext-Base model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
                    return_idx=out_indices, **kwargs)


def convnext_large(out_indices=[1, 2, 3], **kwargs):
    """ ConvNext-Large model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536],
                    return_idx=out_indices, **kwargs)


def convnext_xlarge(out_indices=[1, 2, 3], **kwargs):
    """ ConvNext-XLarge model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048],
                    return_idx=out_indices, **kwargs)


def convnextv2_nano(out_indices=[1, 2, 3], **kwargs):
    """ ConvNextv2-Nano model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXtV2_FPN(
        depths=[2, 2, 8, 2], dims=[80, 160, 320, 640],
        return_idx=out_indices, **kwargs)


def convnextv2_tiny(out_indices=[1, 2, 3], **kwargs):
    """ ConvNextv2-Tiny model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXtV2_FPN(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
        return_idx=out_indices, **kwargs)


def convnextv2_base(out_indices=[1, 2, 3], **kwargs):
    """ ConvNextv2-Base model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXtV2_FPN(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
        return_idx=out_indices, **kwargs)


def convnextv2_large(out_indices=[1, 2, 3], **kwargs):
    """ ConvNextv2-Large model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXtV2_FPN(
        depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536],
        return_idx=out_indices, **kwargs)


def convnextv2_huge(out_indices=[1, 2, 3], **kwargs):
    """ ConvNextv2-Huge model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXtV2_FPN(
        depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816],
        return_idx=out_indices, **kwargs)


convnext_model_dict = {
    'convnext_tiny': convnext_tiny,
    'convnext_small': convnext_small,
    'convnext_base': convnext_base,
    'convnext_large': convnext_large,
    'convnext_xlarge': convnext_xlarge,
    'convnextv2_nano': convnextv2_nano,
    'convnextv2_tiny': convnextv2_tiny,
    'convnextv2_base': convnextv2_base,
    'convnextv2_large': convnextv2_large,
    'convnextv2_huge': convnextv2_huge,
}
