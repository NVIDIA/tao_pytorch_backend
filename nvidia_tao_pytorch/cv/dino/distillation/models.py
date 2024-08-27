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

"""DINO distillation related model definitions."""

from typing import List, Dict, Optional

import torch
import torch.nn as nn

from nvidia_tao_pytorch.core.models import Backbone, TimmBackbone
from nvidia_tao_pytorch.core.distillation.models import LinearNeck, MLPNeck, BackboneNeck


class DinoDistillationStudent(nn.Module):
    """DINO distillation student model."""

    def __init__(self, output_keys: List[str], backbone: Backbone):
        """Initializes the DinoDistillationStudent.

        Args:
            output_keys (List[str]): List of output keys.
            backbone (Backbone): Backbone model.
        """
        super().__init__()
        self.backbone = backbone
        self.output_keys = output_keys

    def out_strides(self):
        """Returns the output strides of the backbone"""
        return self.backbone.out_strides()

    def out_keys(self):
        """Returns the output keys of the backbone"""
        return self.output_keys

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the student model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of output tensors.
        """
        x = self.backbone.forward(x)
        y = {k: v for k, v in zip(self.output_keys, x)}
        return y


def build_dino_distillation_student(model_name: str,
                                    pretrained: bool = True,
                                    neck_type: str = "linear",
                                    mlp_neck_expansion: int = 1,
                                    mlp_neck_depth: int = 1,
                                    neck_kernel_size: int = 1,
                                    mlp_neck_skip_connect: bool = False,
                                    output_keys: List[str] = ['p1', 'p2', 'p3'],
                                    out_channels: List[int] = [256, 384, 768],
                                    pretrained_backbone_path: Optional[str] = None
                                    ):
    """Creates a FAN-S student model.

    Args:
        model_name (str): Name of the model.
        pretrained (bool, optional): If True, uses pre-trained weights. Defaults to True.
        neck_type (str, optional): Type of neck. Defaults to "linear".
        mlp_neck_expansion (int, optional): Expansion factor for MLP neck. Defaults to 1.
        mlp_neck_depth (int, optional): Depth of MLP neck. Defaults to 1.
        neck_kernel_size (int, optional): Kernel size of neck. Defaults to 1.
        mlp_neck_skip_connect (bool, optional): If True, uses skip connections in MLP neck. Defaults to False.
        output_keys (List[str], optional): List of output keys. Defaults to ['p1', 'p2', 'p3'].
        out_channels (List[int], optional): List of output channels. Defaults to [256, 384, 768].

    Returns:
        DinoDistillationStudent: DINO student model.

    """
    backbone = TimmBackbone(
        model_name=model_name,
        pretrained=pretrained,
        out_indices=[-3, -2, -1],
        pretrained_path=pretrained_backbone_path
    )

    if neck_type == "linear":
        neck = LinearNeck(
            in_channels=backbone.out_channels(),
            out_channels=out_channels,
            kernel_size=neck_kernel_size
        )
    elif neck_type == "mlp":
        neck = MLPNeck(
            in_channels=backbone.out_channels(),
            out_channels=out_channels,
            expansion=mlp_neck_expansion,
            depth=mlp_neck_depth,
            kernel_size=neck_kernel_size,
            skip_connect=mlp_neck_skip_connect
        )
    else:
        raise NotImplementedError(f"Neck type {neck_type} not implemented")

    model = DinoDistillationStudent(
        output_keys=output_keys,
        backbone=BackboneNeck(
            backbone=backbone,
            neck=neck
        )
    )

    return model


dino_distillation_student_channels = {
    'resnet34': [128, 256, 512],
    'resnet50': [512, 1024, 2048],
    'efficientvit_b0': [32, 64, 128],
    'efficientvit_b1': [64, 128, 256],
    'efficientvit_b2': [96, 192, 384],
    'efficientvit_b3': [128, 256, 512]
}
