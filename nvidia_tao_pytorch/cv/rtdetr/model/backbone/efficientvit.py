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

"""EfficientViT backbone for RT-DETR. """

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint

from nvidia_tao_pytorch.cv.backbone.nn.ops import (
    ConvLayer,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
    EfficientViTBlock
)

from nvidia_tao_pytorch.cv.backbone.efficientvit import EfficientViTBackbone


class EfficientViT(EfficientViTBackbone):
    """EfficientViT-Base Module"""

    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
        out_indices=[1, 2, 3],
        activation_checkpoint=False,
    ) -> None:
        """Initialize an efficientViT backbone for RT-DETR."""
        super().__init__(
            width_list=width_list,
            depth_list=depth_list,
            in_channels=in_channels,
            dim=dim,
            expand_ratio=expand_ratio,
            norm=norm,
            act_func=act_func
        )
        self.activation_checkpoint = activation_checkpoint
        self.out_channels = width_list[2:]

        self.out_indices = out_indices

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = nn.LayerNorm(width_list[1:][i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    def forward(self, x: torch.Tensor):
        """Forward function."""
        outs = []
        # Stem
        x = self.input_stem(x)

        for idx, stage in enumerate(self.stages):
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = stage(x)
            else:
                x = checkpoint.checkpoint(stage, x)

            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                out = norm_layer(x.permute(0, 2, 3, 1).contiguous())
                outs.append(out.permute(0, 3, 1, 2).contiguous())

        return outs


class EfficientViTLarge(nn.Module):
    """EfficientViT Large Module."""

    def __init__(
        self,
        width_list,
        depth_list,
        block_list=None,
        expand_list=None,
        fewer_norm_list=None,
        in_channels=3,
        qkv_dim=32,
        norm="bn2d",
        act_func="gelu",
        out_indices=[1, 2, 3],
        activation_checkpoint=False,
    ) -> None:
        """Initializing an EfficientViT-Lx series backbone for RT-DETR."""
        super().__init__()
        block_list = block_list or ["res", "fmb", "fmb", "mb", "att"]
        expand_list = expand_list or [1, 4, 4, 4, 6]
        fewer_norm_list = fewer_norm_list or [False, False, False, True, True]

        self.out_indices = out_indices

        self.width_list = []
        self.stages = []
        # stage 0
        stage0 = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                block=block_list[0],
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=expand_list[0],
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[0],
            )
            stage0.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage = []
            block = self.build_local_block(
                block="mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id],
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_list[stage_id] * 4,
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[stage_id],
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                if block_list[stage_id].startswith("att"):
                    stage.append(
                        EfficientViTBlock(
                            in_channels=in_channels,
                            dim=qkv_dim,
                            expand_ratio=expand_list[stage_id],
                            scales=(3,) if block_list[stage_id] == "att@3" else (5,),
                            norm=norm,
                            act_func=act_func,
                        )
                    )
                else:
                    block = self.build_local_block(
                        block=block_list[stage_id],
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        expand_ratio=expand_list[stage_id],
                        norm=norm,
                        act_func=act_func,
                        fewer_norm=fewer_norm_list[stage_id],
                    )
                    block = ResidualBlock(block, IdentityLayer())
                    stage.append(block)
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

        self.out_channels = width_list[2:]

        self.activation_checkpoint = activation_checkpoint
        # add a norm layer for each output
        for i_layer in out_indices:
            layer = nn.LayerNorm(width_list[1:][i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    @staticmethod
    def build_local_block(
        block: str,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        """Build local block."""
        if block == "res":
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "fmb":
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "mb":
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        else:
            raise ValueError(block)
        return block

    def forward(self, x: torch.Tensor):
        """Forward function."""
        outs = []
        # Stem
        x = self.stages[0](x)

        for idx, stage in enumerate(self.stages[1:]):
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = stage(x)
            else:
                x = checkpoint.checkpoint(stage, x)

            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                out = norm_layer(x.permute(0, 2, 3, 1).contiguous())
                outs.append(out.permute(0, 3, 1, 2).contiguous())

        return outs


def efficientvit_b0(out_indices=[1, 2, 3], **kwargs):
    """ EfficientViT-B0 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViT(width_list=[8, 16, 32, 64, 128],
                        depth_list=[1, 2, 2, 2, 2],
                        dim=16,
                        out_indices=out_indices, **kwargs)


def efficientvit_b1(out_indices=[1, 2, 3], **kwargs):
    """ EfficientViT-B1 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViT(width_list=[16, 32, 64, 128, 256],
                        depth_list=[1, 2, 3, 3, 4],
                        dim=16,
                        out_indices=out_indices, **kwargs)


def efficientvit_b2(out_indices=[1, 2, 3], **kwargs):
    """ EfficientViT-B2 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViT(width_list=[24, 48, 96, 192, 384],
                        depth_list=[1, 3, 4, 4, 6],
                        dim=32,
                        out_indices=out_indices, **kwargs)


def efficientvit_b3(out_indices=[1, 2, 3], **kwargs):
    """ EfficientViT-B3 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViT(width_list=[32, 64, 128, 256, 512],
                        depth_list=[1, 4, 6, 6, 9],
                        dim=32,
                        out_indices=out_indices, **kwargs)


def efficientvit_l0(out_indices=[1, 2, 3], **kwargs):
    """ EfficientViT-L0 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViTLarge(width_list=[32, 64, 128, 256, 512],
                             depth_list=[1, 1, 1, 4, 4],
                             out_indices=out_indices, **kwargs)


def efficientvit_l1(out_indices=[1, 2, 3], **kwargs):
    """ EfficientViT-L1 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViTLarge(width_list=[32, 64, 128, 256, 512],
                             depth_list=[1, 1, 1, 6, 6],
                             out_indices=out_indices, **kwargs)


def efficientvit_l2(out_indices=[1, 2, 3], **kwargs):
    """ EfficientViT-L2 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViTLarge(width_list=[32, 64, 128, 256, 512],
                             depth_list=[1, 2, 2, 8, 8],
                             out_indices=out_indices, **kwargs)


def efficientvit_l3(out_indices=[1, 2, 3], **kwargs):
    """ EfficientViT-L3 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViTLarge(width_list=[64, 128, 256, 512, 1024],
                             depth_list=[1, 2, 2, 8, 8],
                             out_indices=out_indices, **kwargs)


efficientvit_model_dict = {
    "efficientvit_b0": efficientvit_b0,
    "efficientvit_b1": efficientvit_b1,
    "efficientvit_b2": efficientvit_b2,
    "efficientvit_b3": efficientvit_b3,
    "efficientvit_l0": efficientvit_l0,
    "efficientvit_l1": efficientvit_l1,
    "efficientvit_l2": efficientvit_l2,
    "efficientvit_l3": efficientvit_l3,
}
