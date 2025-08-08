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

# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

"""EfficientViT backbone.

Based on the TAO team's experience, the TIMM implementation doesn't match
the performance of the original implementaion. Therefore, we stick to the
original implementation available at:
https://github.com/mit-han-lab/efficientvit
"""

from functools import partial
from inspect import signature
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


################################################################################
#                             Utility                                          #
################################################################################


def val2list(x, repeat_time=1):
    """Value to list."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x, min_len=1, idx_repeat=-1):
    """Value to tuple."""
    x = val2list(x)
    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)


def build_kwargs_from_config(config: dict, target_func: callable) -> dict[str, any]:
    """Build kwargs from config."""
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def build_act(name: str, **kwargs):
    """Build activation layer."""
    REGISTERED_ACT_DICT: dict[str, type] = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "hswish": nn.Hardswish,
        "silu": nn.SiLU,
        "gelu": partial(nn.GELU, approximate="tanh"),
    }
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]):
    """Get same padding."""
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def resize(x, size=None, scale_factor=None, mode="bicubic", align_corners=False):
    """Resize tensor."""
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


################################################################################
#                             Basic Layers                                     #
################################################################################


class ConvLayer(nn.Module):
    """Conv layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super().__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        if norm == "bn2d":
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm {norm} not implemented.")
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor):
        """Forward."""
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    """Upsample layer."""

    def __init__(
        self,
        mode="bicubic",
        size=None,
        factor=2,
        align_corners=False,
    ):
        super().__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor):
        """Forward."""
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


################################################################################
#                             Basic Blocks                                     #
################################################################################


class DSConv(nn.Module):
    """DS Conv."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor):
        """Forward."""
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    """MB Conv."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    """Fused MB Conv layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ResBlock(nn.Module):
    """ResBlock."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=None,
        heads_ratio=1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales=(5,),
        eps=1.0e-15,
    ):
        super().__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(in_channels, 3 * total_dim, 1, use_bias=use_bias[0], norm=norm[0], act_func=act_func[0])
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)), out_channels, 1, use_bias=use_bias[1], norm=norm[1], act_func=act_func[1]
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor):
        """ReLU Linear attention."""
        B, _, H, W = list(qkv.size())
        if qkv.dtype == torch.float16:
            qkv = qkv.float()
        qkv = torch.reshape(
            qkv,
            (B, -1, 3 * self.dim, H * W),
        )
        q, k, v = (
            qkv[:, :, 0: self.dim],
            qkv[:, :, self.dim: 2 * self.dim],
            qkv[:, :, 2 * self.dim:],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_quadratic_att(self, qkv: torch.Tensor):
        """ReLU Quadratic attention."""
        B, _, H, W = list(qkv.size())
        qkv = torch.reshape(
            qkv,
            (B, -1, 3 * self.dim, H * W),
        )
        q, k, v = (
            qkv[:, :, 0: self.dim],
            qkv[:, :, self.dim: 2 * self.dim],
            qkv[:, :, 2 * self.dim:],
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)  # b h n n
        original_dtype = att_map.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)  # b h n n
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)  # b h d n
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor):
        """Forward."""
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)
        H, W = list(qkv.size())[-2:]
        if H * W > self.dim:
            out = self.relu_linear_att(qkv)
        else:
            out = self.relu_quadratic_att(qkv)
        out = self.proj(out)
        return out


class EfficientViTBlock(nn.Module):
    """EfficientViT Block."""

    def __init__(
        self, in_channels, heads_ratio=1.0, dim=32, expand_ratio=4, scales=(5,), norm="bn2d", act_func="hswish"
    ):
        super().__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                scales=scales,
            ),
            nn.Identity(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, nn.Identity())

    def forward(self, x: torch.Tensor):
        """Forward."""
        x = self.context_module(x)
        x = self.local_module(x)
        return x


################################################################################
#                             Functional Blocks                                #
################################################################################


class ResidualBlock(nn.Module):
    """Residual Block."""

    def __init__(self, main, shortcut, post_act=None, pre_norm=None):
        super().__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor):
        """Forward main."""
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        return self.main(x)

    def forward(self, x: torch.Tensor):
        """Forward."""
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class OpSequential(nn.Module):
    """Op Sequential."""

    def __init__(self, op_list):
        super().__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor):
        """Forward."""
        for op in self.op_list:
            x = op(x)
        return x


################################################################################
#                             Backbones                                        #
################################################################################


class EfficientViT(BackboneBase):
    """EfficientViT model.

    EfficientViT is a new family of high-resolution vision models with novel multi-scale linear attention. Unlike prior
    high-resolution dense prediction models that rely on heavy softmax attention, hardware-inefficient large-kernel
    convolution, or complicated topology structure to obtain good performances, EfficientViT uses multi-scale linear
    attention achieves the global receptive field and multi-scale learning with only lightweight and hardware-efficient
    operations.

    References:
    - [EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction](
      https://arxiv.org/abs/2205.14756)
    - [https://github.com/mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)
    """

    def __init__(
        self,
        in_chans=3,
        width_list=(8, 16, 32, 64, 128),
        depth_list=(1, 2, 2, 2, 2),
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
        num_classes=1000,
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        **kwargs,
    ):
        """Initialize the EfficientViT model.

        Args:
            in_chans: Number of input image channels.
            width_list: Feature dimension at each stage.
            depth_list: Number of blocks at each stage.
            dim: Dimension of the head.
            expand_ratio: Expand ratio for the MBConv block.
            norm: Normalization layer type.
            act_func: Activation layer type.
            num_classes: Number of classes for classification head.
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

        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=in_chans,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, nn.Identity()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, nn.Identity() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

        self.num_features = in_channels
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        """Build local block."""
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {0: self.input_stem}
        for i, stage in enumerate(self.stages, start=1):
            stage_dict[i] = stage
        return stage_dict

    def get_classifier(self):
        """Get the classifier module."""
        return self.head

    def reset_classifier(self, num_classes=0, **kwargs):
        """Reset the classifier head."""
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head."""
        x = self.input_stem(x)
        for stage in self.stages:
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = stage(x)
            else:
                x = checkpoint.checkpoint(stage, x)
        return x.mean(dim=[2, 3])

    def forward_feature_pyramid(self, *args, **kwargs):
        """Forward pass through the backbone to extract intermediate feature maps."""
        raise NotImplementedError("forward_feature_pyramid is not implemented.")

    def forward(self, x):
        """Forward."""
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


class EfficientViTLarge(BackboneBase):
    """EfficientViT Large model.

    EfficientViT is a new family of high-resolution vision models with novel multi-scale linear attention. Unlike prior
    high-resolution dense prediction models that rely on heavy softmax attention, hardware-inefficient large-kernel
    convolution, or complicated topology structure to obtain good performances, EfficientViT uses multi-scale linear
    attention achieves the global receptive field and multi-scale learning with only lightweight and hardware-efficient
    operations.

    References:
    - [EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction](
      https://arxiv.org/abs/2205.14756)
    - [https://github.com/mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)
    """

    def __init__(
        self,
        in_chans=3,
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        qkv_dim=32,
        norm="bn2d",
        act_func="gelu",
        num_classes=1000,
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        **kwargs,
    ):
        """Initialize the EfficientViTLarge class.

        Args:
            in_chans (int): Number of input image channels. Default: `3`.
            width_list: Feature dimension at each stage.
            depth_list: Number of blocks at each stage.
            qkv_dim: Dimension of the head.
            norm: Normalization layer type.
            act_func: Activation layer type.
            num_classes (int): Number of classes for classification head. Default: `1000`.
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

        self.width_list = []
        self.stages = []
        # stage 0
        stage0 = [ConvLayer(in_channels=in_chans, out_channels=width_list[0], stride=2, norm=norm, act_func=act_func)]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                stage_id=0,
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            stage0.append(ResidualBlock(block, nn.Identity()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[1:4], depth_list[1:4]), start=1):
            stage = []
            for i in range(d + 1):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    stage_id=stage_id,
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=4 if stride == 1 else 16,
                    norm=norm,
                    act_func=act_func,
                    fewer_norm=stage_id > 2,
                )
                block = ResidualBlock(block, nn.Identity() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[4:], depth_list[4:]), start=4):
            stage = []
            block = self.build_local_block(
                stage_id=stage_id,
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=24,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=qkv_dim,
                        expand_ratio=6,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

        self.num_features = in_channels
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @staticmethod
    def build_local_block(
        stage_id: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        """Build local block."""
        if expand_ratio == 1:
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif stage_id <= 2:
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {}
        for i, stage in enumerate(self.stages):
            stage_dict[i] = stage
        return stage_dict

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classifier module."""
        return self.head

    def reset_classifier(self, num_classes=0, **kwargs):
        """Reset the classifier head."""
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head."""
        for stage in self.stages:
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = stage(x)
            else:
                x = checkpoint.checkpoint(stage, x)
        return x

    def forward_feature_pyramid(self, *args, **kwargs):
        """Forward pass through the backbone to extract intermediate feature maps."""
        raise NotImplementedError("forward_feature_pyramid is not implemented.")

    def forward(self, x):
        """Forward."""
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def efficientvit_b0(**kwargs):
    """Constructs a EfficientViT-B0 model."""
    return EfficientViT(width_list=[8, 16, 32, 64, 128], depth_list=[1, 2, 2, 2, 2], dim=16, **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_b1(**kwargs):
    """Constructs a EfficientViT-B1 model."""
    return EfficientViT(width_list=[16, 32, 64, 128, 256], depth_list=[1, 2, 3, 3, 4], dim=16, **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_b2(**kwargs):
    """Constructs a EfficientViT-B2 model."""
    return EfficientViT(width_list=[24, 48, 96, 192, 384], depth_list=[1, 3, 4, 4, 6], dim=32, **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_b3(**kwargs):
    """Constructs a EfficientViT-B3 model."""
    return EfficientViT(width_list=[32, 64, 128, 256, 512], depth_list=[1, 4, 6, 6, 9], dim=32, **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_l0(**kwargs):
    """Constructs a EfficientViT-L0 model."""
    return EfficientViTLarge(width_list=[32, 64, 128, 256, 512], depth_list=[1, 1, 1, 4, 4], **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_l1(**kwargs):
    """Constructs a EfficientViT-L1 model."""
    return EfficientViTLarge(width_list=[32, 64, 128, 256, 512], depth_list=[1, 1, 1, 6, 6], **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_l2(**kwargs):
    """Constructs a EfficientViT-L2 model."""
    return EfficientViTLarge(width_list=[32, 64, 128, 256, 512], depth_list=[1, 2, 2, 8, 8], **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_l3(**kwargs):
    """Constructs a EfficientViT-L3 model."""
    return EfficientViTLarge(width_list=[64, 128, 256, 512, 1024], depth_list=[1, 2, 2, 8, 8], **kwargs)
