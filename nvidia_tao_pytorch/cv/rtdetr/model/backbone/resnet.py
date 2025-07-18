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

""" Backbone ResNet model definition. """

import torch.nn as nn

from nvidia_tao_pytorch.cv.deformable_detr.model.resnet import BasicBlock, Bottleneck, conv1x1
from nvidia_tao_pytorch.cv.deformable_detr.model.backbone import FrozenBatchNorm2d


class ResNet(nn.Module):
    """ Baset ResNet Module class """

    def __init__(self, layers, block=Bottleneck, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 freeze_norm=True, freeze_at=-1, num_stages=4, return_idx=[0, 1, 2, 3]):
        """Init"""
        super(ResNet, self).__init__()

        self.inplanes = 64

        if freeze_norm:
            self._norm_layer = FrozenBatchNorm2d
        else:
            self._norm_layer = nn.BatchNorm2d

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.return_idx = return_idx
        out_channels = [256, 512, 1024, 2048]
        if block == BasicBlock:
            out_channels = [64, 128, 256, 512]

        _out_strides = [4, 8, 16, 32]
        self.out_channels = [out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Make_layer"""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for idx, stage in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


def resnet18(out_indices=[1, 2, 3], **kwargs):
    """ Resnet-18 model from
        Deep Residual Learning for Image Recognition
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                  freeze_at=0, return_idx=out_indices,
                  replace_stride_with_dilation=[False, False, False],
                  freeze_norm=True, **kwargs)


def resnet34(out_indices=[1, 2, 3], **kwargs):
    """ Resnet-34 model from
        Deep Residual Learning for Image Recognition
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ResNet(block=BasicBlock, layers=[3, 4, 6, 3],
                  freeze_at=0, return_idx=out_indices,
                  replace_stride_with_dilation=[False, False, False],
                  freeze_norm=True, **kwargs)


def resnet50(out_indices=[1, 2, 3], **kwargs):
    """ ResNet-50 model from
        Deep Residual Learning for Image Recognition

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3],
                  freeze_at=0, return_idx=out_indices,
                  replace_stride_with_dilation=[False, False, False],
                  freeze_norm=True, **kwargs)


def resnet101(out_indices=[1, 2, 3], **kwargs):
    """ ResNet-101 model from
        Deep Residual Learning for Image Recognition

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ResNet(block=Bottleneck, layers=[3, 4, 23, 3],
                  freeze_at=0, return_idx=out_indices,
                  replace_stride_with_dilation=[False, False, False],
                  freeze_norm=True, **kwargs)


resnet_model_dict = {
    'resnet_18': resnet18,
    'resnet_34': resnet34,
    'resnet_50': resnet50,
    'resnet_101': resnet101,
}
