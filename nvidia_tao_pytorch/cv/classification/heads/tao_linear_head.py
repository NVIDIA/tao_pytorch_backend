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

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from mmpretrain.models.heads import ClsHead
from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.structures import DataSample

from typing import List, Tuple


@MODELS.register_module()
class TAOLinearClsHead(ClsHead):
    """Linear classifier head updated from MMPretrain to fix Feat return Bug.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 binary,
                 num_classes,
                 in_channels,
                 head_init_scale=None,
                 init_cfg=None,
                 classifier=None,
                 *args, # noqa pylint: disable=W1113
                 **kwargs # noqa pylint: disable=W1113
                 ):
        """ Init Module """
        super(TAOLinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.head_init_scale = head_init_scale
        self.binary = binary

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        if self.num_classes != 2 and self.binary:
            raise ValueError(
                f'Only support binary head when num_classes == 2, Got num_classes == {self.num_classes}'
            )

        if self.binary:
            self.fc = nn.Linear(self.in_channels, 1)
        else:
            self.fc = nn.Linear(self.in_channels, self.num_classes)
        if head_init_scale:
            self.fc.weight.data.mul_(head_init_scale)
            self.fc.bias.data.mul_(head_init_scale)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)

        return cls_score

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.
        Overwrite the _get_predictions for binary classification purpose
        """
        if self.binary:
            # default threshold set to 0.5 for binary classification
            pred_scores = F.sigmoid(cls_score)

            # transform binary to two neurons output to fit the format requirement in
            # Inferencer visualization method - self.visualizer.visualize_cls
            pred_scores = torch.cat((1.0 - pred_scores, pred_scores), dim=1)

            # pred_labels = pred_scores.gt(0.5).int().detach()
        else:
            pred_scores = F.softmax(cls_score, dim=1)

        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)

        return out_data_samples
