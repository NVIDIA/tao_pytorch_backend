# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Original source taken from https://github.com/open-mmlab/mmsegmentation

# Copyright 2019 OpenMMLAB

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" FAN Linear Class Head """

import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads import ClsHead


@HEADS.register_module()
class FANLinearClsHead(ClsHead):
    """Linear classifier head.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 head_init_scale,
                 backbone=None,
                 init_cfg=None,
                 *args, # noqa pylint: disable=W1113
                 **kwargs # noqa pylint: disable=W1113
                 ):
        """ Init Module """
        super(FANLinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.head_init_scale = head_init_scale

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.fc.weight.data.mul_(head_init_scale)
        self.fc.bias.data.mul_(head_init_scale)

    def pre_logits(self, x):
        """ Function to Get the Features """
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.
        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.
        Returns:
            Tensor | list: The inference results.
                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.fc(x)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score
        if post_process:
            return self.post_process(pred)
        return pred

    def forward_train(self, x, gt_label, **kwargs):
        """ Forward Train Module """
        x = self.pre_logits(x)
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses
