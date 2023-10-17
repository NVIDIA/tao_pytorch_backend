# Copyright (c) 2023 Chaminda Bandara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Original source taken from https://github.com/wgcban/ChangeFormer
#
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

"""Visual ChangeNet Segmentation model builder"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.changenet_utils import (
    MLP, conv_diff, make_prediction, UpsampleConvLayer,
    ResidualBlock, ConvLayer, resize, count_params,
)
from nvidia_tao_pytorch.cv.visual_changenet.backbone.fan import fan_model_dict
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import get_global_rank, load_pretrained_weights

logger = logging.getLogger(__name__)


class DecoderTransformer_v3(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[32, 64, 128, 256], embedding_dim=64, output_nc=2,
                 decoder_softmax=False, feature_strides=[2, 4, 8, 16], model_name='FANHybrid'):
        """Initialize DecoderTransformer_v3 class

        Args:
            input_transform (str): Input transformation strategy.
            in_index (list): Indices of input features to use.
            align_corners (bool): Align corners flag for resizing.
            in_channels (list): Input channels for each feature level.
            embedding_dim (int): Embedding dimension for decoder.
            output_nc (int): Number of output classes.
            feature_strides (list): Feature stride for each level.
            model_name (str): Backbone model name.
            decoder_softmax (bool): Flag to indicate whether to use softmax for predictions
        """
        super(DecoderTransformer_v3, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels), f"Feature strides {feature_strides} is not equal to the in_channels {in_channels}"
        assert min(feature_strides) == feature_strides[0], f"First feature stride from {feature_strides} is not the minimum {min(feature_strides)}"

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        self.model_name = model_name

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        # convolutional Difference Modules
        self.diff_c4 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        # taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        """
        Forward pass of the Visual ChangeNetSegment decoder.
        """
        # Transforming encoder features (select layers)
        # Just takes the 4 encoder feature maps as give in comment below
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        # MLP decoder on C1-C4
        n, _, _, _ = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        p_c4 = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        if self.feature_strides[-1] == self.feature_strides[-2]:
            _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + _c4
        else:
            _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")

        p_c3 = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2 = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1 = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


# Visual ChangeNetSegment:
class ChangeNetSegment(nn.Module):
    """
    Visual ChangeNetSegment model for semantic segmentation using a Transformer Encoder and Decoder.

    Args:
        input_nc (int): Number of input channels (default is 3).
        output_nc (int): Number of output classes (default is 2).
        decoder_softmax (bool): Whether to apply softmax to the decoder output (default is False).
        embed_dim (int): Embedding dimension for the Transformer Encoder (default is 256).
        model (str): Backbone model name (default is 'fan_tiny_8_p4_hybrid_256').
        img_size (int): Input image size (default is 256).
        embed_dims (list): List of embedding dimensions for the Transformer Decoder (default is [128, 256, 384, 384]).
        feature_strides (list): List of feature strides for the Transformer Decoder (default is [4, 8, 16, 16]).
        in_index (list): List of input indices for the Transformer Decoder (default is [0, 1, 2, 3]).
        feat_downsample: Flag to indicate whether to use feature downsampling for the transformer block for FAN-Hybrid
    """

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256, model='fan_tiny_8_p4_hybrid_256', img_size=256,
                 embed_dims=[128, 256, 384, 384], feature_strides=[4, 8, 16, 16], in_index=[0, 1, 2, 3], feat_downsample=False,
                 pretrained_backbone_path=None):
        """Initialize Visual ChangeNetSegment class"""
        super(ChangeNetSegment, self).__init__()
        # Transformer Encoder

        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1
        self.model_name = model
        self.embed_dims = embed_dims

        logger.info(f"Number of output classes: {output_nc}")
        assert img_size % feature_strides[-1] == 0, 'Input image size must be a multiple of 16'
        # TODO: @zbhat - support FANSwin?
        self.backbone = fan_model_dict[self.model_name](
            pretrained=False,
            num_classes=output_nc,
            checkpoint_path='',
            img_size=img_size,
            feat_downsample=feat_downsample)

        # missing_keys = None
        if pretrained_backbone_path:
            checkpoint = load_pretrained_weights(pretrained_backbone_path)
            _tmp_st_output = self.backbone.load_state_dict(checkpoint, strict=False)
            # missing_keys = list(_tmp_st_output[0])
            if get_global_rank() == 0:
                logger.info(f"Loaded pretrained weights from {pretrained_backbone_path}")
                logger.info(f"{_tmp_st_output}")

        # Transformer Decoder
        self.decoder = DecoderTransformer_v3(input_transform='multiple_select', in_index=in_index, align_corners=False,
                                             in_channels=self.embed_dims, embedding_dim=self.embedding_dim, output_nc=output_nc,
                                             decoder_softmax=decoder_softmax, feature_strides=feature_strides, model_name=self.model_name)

    def forward(self, x1, x2):
        """
        Forward pass of the Visual ChangeNetSegment model.

        Args:
            x1 (torch.Tensor): Input tensor for the first image input.
            x2 (torch.Tensor): Input tensor for the second image input.

        Returns:
            torch.Tensor: Output tensor representing the segmentation map.
        """
        [fx1, fx2] = [self.backbone(x1), self.backbone(x2)]
        out_decoder = self.decoder(fx1, fx2)
        return out_decoder


def build_model(experiment_config,
                export=False):
    """ Build changenet model according to configuration

    Args:
        experiment_config: experiment configuration
        export: flag to indicate onnx export

    Returns:
        model

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset.segment

    backbone = model_config.backbone['type']
    feat_downsample = model_config.backbone.feat_downsample
    pretrained_backbone_path = model_config.backbone.pretrained_backbone_path

    channels_map = {"fan_tiny_8_p4_hybrid": [128, 256, 192, 192],
                    "fan_large_16_p4_hybrid": [128, 256, 480, 480],
                    "fan_small_12_p4_hybrid": [128, 256, 384, 384],
                    "fan_base_16_p4_hybrid": [128, 256, 448, 448], }
    if backbone in channels_map:
        embed_dims = channels_map[backbone]
        model_config.decode_head.in_channels = embed_dims

    else:
        raise NotImplementedError('Bacbkbone name [%s] is not supported' % backbone)

    embed_dim = model_config.decode_head.decoder_params['embed_dim']
    feature_strides = model_config.decode_head.feature_strides
    in_index = model_config.decode_head.in_index

    num_classes = dataset_config.num_classes
    img_size = dataset_config.img_size

    model = ChangeNetSegment(embed_dim=embed_dim,
                             model=backbone,
                             output_nc=num_classes,
                             img_size=img_size,
                             input_nc=3,
                             decoder_softmax=False,
                             embed_dims=embed_dims,
                             feature_strides=feature_strides,
                             in_index=in_index,
                             feat_downsample=feat_downsample,
                             pretrained_backbone_path=pretrained_backbone_path)
    count_params(model)

    return model
