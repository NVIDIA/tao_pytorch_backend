# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Original source taken from https://github.com/michuanhaohao/reid-strong-baseline
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

"""Baseline Module for Re-Identification."""

import torch
from torch import nn
from nvidia_tao_pytorch.cv.re_identification.model.resnet import Bottleneck, ResNet, BasicBlock


def weights_init_kaiming(m):
    """Initializes weights using Kaiming Normal initialization.

    Args:
        m (torch.nn.Module): PyTorch module whose weights are to be initialized.

    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """Initializes the weights of a classifier layer.

    Args:
        m (torch.nn.Module): PyTorch module whose weights are to be initialized.

    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    """Baseline model for re-identification tasks.

    This class generates a model based on the provided configuration. The model
    is primarily a ResNet variant, with additional features like bottleneck and classifier
    layers. The ResNet architecture can be one of the following variants: 18, 34, 50, 101, 152.

    Attributes:
        in_planes (int): Dimensionality of the input features.
        base (ResNet): Base ResNet model.
        gap (torch.nn.AdaptiveAvgPool2d): Global Average Pooling layer.
        num_classes (int): Number of output classes.
        neck (str): Specifies the neck architecture of the model.
        neck_feat (str): Specifies whether neck features are used.
        if_flip_feat (bool): Whether to flip the features or not.
        classifier (torch.nn.Linear): Classifier layer of the model.
        bottleneck (torch.nn.BatchNorm1d): Optional bottleneck layer of the model.
    """

    def __init__(self, cfg, num_classes):
        """Initializes the Baseline model with provided configuration and number of classes.

        Args:
            cfg (DictConfig): Configuration object containing model parameters.
            num_classes (int): Number of output classes.
        """
        super(Baseline, self).__init__()
        self.in_planes = cfg['model']['feat_dim']
        if "resnet" in cfg['model']['backbone']:

            arch_settings = {
                'resnet_18': (BasicBlock, [2, 2, 2, 2]),
                'resnet_34': (BasicBlock, [3, 4, 6, 3]),
                'resnet_50': (Bottleneck, [3, 4, 6, 3]),
                'resnet_101': (Bottleneck, [3, 4, 23, 3]),
                'resnet_152': (Bottleneck, [3, 8, 36, 3])
            }

            self.base = ResNet(feat_dim=cfg['model']['feat_dim'], last_stride=cfg['model']['last_stride'],
                               block=Bottleneck,
                               layers=arch_settings[cfg['model']['backbone']][1])

        if cfg['model']['pretrain_choice'] == 'imagenet':
            if cfg['model']['pretrained_model_path']:
                self.base.load_param(cfg['model']['pretrained_model_path'])
                print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = cfg['model']['neck']
        self.neck_feat = cfg['model']['neck_feat']
        self.if_flip_feat = cfg['model']['with_flip_feature']

        if not self.neck:
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        """Defines the forward pass of the Baseline model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after forward pass. This could be feature embeddings
            or the sum of feature embeddings in case of flipped features.
        """
        if self.training:
            return self.__forward(x)
        if self.if_flip_feat:
            y = torch.flip(x, [3])
            feat1 = self.__forward(y)
            feat2 = self.__forward(x)
            return feat2 + feat1
        return self.__forward(x)

    def __forward(self, x):
        """Internal method for processing the features through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing. This could be the class scores
            and global features during training or the feature embeddings during testing.
        """
        global_feat = self.gap(self.base(x))
        global_feat = global_feat.view(global_feat.shape[0], -1)

        if not self.neck:
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        if self.neck_feat == 'after':
            # cls_score = self.classifier(feat)
            return feat
            # return cls_score, global_feat  # global feature for triplet loss
        return global_feat
