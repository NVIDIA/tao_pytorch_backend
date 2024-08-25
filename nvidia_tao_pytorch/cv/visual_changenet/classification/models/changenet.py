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

"""Visual ChangeNet Classification model builder"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.changenet_utils import (
    MLP, conv_diff, resize, count_params,
)
from nvidia_tao_pytorch.cv.visual_changenet.backbone.fan import fan_model_dict
from nvidia_tao_pytorch.cv.visual_changenet.backbone.vision_transformer.dinov2_vit import vit_model_dict
from nvidia_tao_pytorch.cv.visual_changenet.backbone.vision_transformer.vit_adapter import vit_adapter_model_dict
from nvidia_tao_pytorch.cv.visual_changenet.utils.pos_embed_interpolation_converter import (
    interpolate_patch_embed, interpolate_pos_embed
)
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import load_pretrained_weights
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
logger = logging.getLogger(__name__)


class ChangeNetClassifyDecoder(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[32, 64, 128, 256], embedding_dim=64, output_nc=2,
                 feature_strides=[2, 4, 8, 16], model_name='FANHybrid',
                 num_input=4, output_shape=[128, 128], embed_dec=30, learnable_difference_modules=4):
        """
        Initialize Visual ChangeNetClassifyDecoder.

        Args:
            input_transform (str): Input transformation strategy.
            in_index (list): Indices of input features to use.
            align_corners (bool): Align corners flag for resizing.
            in_channels (list): Input channels for each feature level.
            embedding_dim (int): Embedding dimension for decoder.
            output_nc (int): Number of output classes.
            feature_strides (list): Feature stride for each level.
            model_name (str): Backbone model name.
            num_input (int): Number of input image lighting conditions.
            output_shape (list): Output shape of the model.
            embed_dec (int): Embedding dimension for the final classifier.
            learnable_difference_modules (int): Number of decoder difference modules for Architecture 2.
        """
        super(ChangeNetClassifyDecoder, self).__init__()
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
        self.model_name = model_name
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        self.learnable_difference_modules = learnable_difference_modules

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

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(nn.Conv2d(in_channels=self.embedding_dim * learnable_difference_modules,
                                                   out_channels=self.embedding_dim, kernel_size=1),
                                         nn.BatchNorm2d(self.embedding_dim)
                                         )

        self.active = nn.Sigmoid()
        # Image level classification

        self.dim_output = output_shape[0] // self.feature_strides[0]
        self.dim_output1 = (output_shape[1] * num_input) // self.feature_strides[0]
        self.in_channel_dim = self.embedding_dim * self.dim_output * self.dim_output1

        self.classifier = nn.Sequential(
            nn.Linear(self.in_channel_dim, embed_dec),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dec),
            nn.Linear(embed_dec, embed_dec),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dec),
            nn.Linear(embed_dec, self.output_nc)
        )

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
        Forward pass of the Visual ChangeNetClassifyDecoder.
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

        # Stage 4: x1/16 scale
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        if self.feature_strides[-2] == self.feature_strides[-1]:
            _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + _c4
        else:
            _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")

        difference_modules = tuple([_c4_up, _c3_up, _c2_up, _c1][:self.learnable_difference_modules])
        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat(difference_modules, dim=1))
        # For Visual ChangeNet Classifier
        _c_flatten = _c.reshape(_c.size(0), -1)
        output = self.classifier(_c_flatten)

        return output


# Visual ChangeNetClassify:
class ChangeNetClassify(nn.Module):
    """
    Visual ChangeNetClassify model for change classification using a Transformer Encoder and Decoder.

    Args:
        input_nc (int): Number of input channels (default is 3).
        output_nc (int): Number of output classes (default is 2).
        embed_dim (int): Embedding dimension for the Transformer Encoder (default is 256).
        model (str): Backbone model name (default is 'fan_tiny_8_p4_hybrid_256').
        embed_dims (list): List of embedding dimensions for the Transformer Decoder (default is [128, 256, 384, 384]).
        feature_strides (list): List of feature strides for the Transformer Decoder (default is [4, 8, 16, 16]).
        in_index (list): List of input indices for the Transformer Decoder (default is [0, 1, 2, 3]).
        num_input: Number of input lighting conditions combined for each input image
        output_shape: Image input shape to the model as output by the dataloader after augmentation
        embedding_vectors: For architecture 1, the dimension for the last embedding vector for each input image for computing eulidean distance
        embed_dec: For architecture 2, the embedding dimension of the output MLP.
        feat_downsample: Flag to indicate whether to use feature downsampling for the transformer block for FAN-Hybrid
        learnable_difference_modules (int): Number of decoder difference modules for Architecture 2
        pretrained_backbone_path (str): Path to the pre-trained backbone weights.
        activation_checkpoint (bool): Enable activation checkpointing
        freeze_backbone: Flag to freeze backbone weights during training.
        return_interm_indices (list): list of layer indices to reutrn as backbone features.
    """

    def __init__(self, input_nc=3, output_nc=2, embed_dim=256, model='fan_tiny_8_p4_hybrid_256',
                 embed_dims=[128, 256, 384, 384], feature_strides=[4, 8, 16, 16], in_index=[0, 1, 2, 3],
                 difference_module='learnable', num_input=1, output_shape=[128, 128], embedding_vectors=5, embed_dec=30,
                 feat_downsample=False, return_interm_indices=[0, 1, 2, 3], learnable_difference_modules=4, pretrained_backbone_path=None, activation_checkpoint=False, freeze_backbone=False):
        """Initialize Visual ChangeNetSegment class"""
        super(ChangeNetClassify, self).__init__()

        # Index 4 is not part of the backbone but taken from index 3 with conv 3x3 stride 2
        return_interm_indices = [r for r in return_interm_indices if r != 4]
        return_interm_indices = np.array(return_interm_indices)
        if not np.logical_and(return_interm_indices >= 0, return_interm_indices <= 4).all():
            raise ValueError(f"Invalid range for return_interm_indices. "
                             f"Provided return_interm_indices is {return_interm_indices}.")

        if len(np.unique(return_interm_indices)) != len(return_interm_indices):
            raise ValueError(f"Duplicate index in the provided return_interm_indices: {return_interm_indices}")

        # Transformer Encoder
        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1
        self.model_name = model
        self.embed_dims = embed_dims
        self.difference_module = difference_module

        logger.info(f"Number of output classes: {output_nc}")

        pretrained_backbone_ckp = load_pretrained_weights(pretrained_backbone_path) if pretrained_backbone_path else None

        if 'fan' in self.model_name:
            assert (output_shape[0] % feature_strides[-1] == 0) and (output_shape[1] % feature_strides[-1] == 0), 'Input image size must be a multiple of 16'
            self.backbone = fan_model_dict[self.model_name](
                pretrained=False,
                num_classes=output_nc,
                checkpoint_path='',
                feat_downsample=feat_downsample)

        elif 'vit' in self.model_name:
            assert output_shape[0] == output_shape[1], 'ViT Backbones only support square input image where input_width == input_height'
            if self.difference_module == 'learnable':
                self.backbone = vit_adapter_model_dict[self.model_name](
                    out_indices=return_interm_indices,
                    resolution=output_shape[0],
                    activation_checkpoint=activation_checkpoint)

                # do interpolation
                pretrained_backbone_ckp = interpolate_vit_checkpoint(checkpoint=pretrained_backbone_ckp,
                                                                     target_patch_size=16,
                                                                     target_resolution=output_shape[0])
            elif self.difference_module == 'euclidean':
                self.backbone = vit_model_dict[self.model_name]()

                # do interpolation
                pretrained_backbone_ckp = interpolate_vit_checkpoint(checkpoint=pretrained_backbone_ckp,
                                                                     target_patch_size=14,
                                                                     target_resolution=518)

        else:
            raise NotImplementedError('Bacbkbone name [%s] is not supported' % self.model_name)

        if pretrained_backbone_ckp:
            _tmp_st_output = self.backbone.load_state_dict(pretrained_backbone_ckp, strict=False)

            if get_global_rank() == 0:
                logger.info(f"Loaded pretrained weights from {pretrained_backbone_path}")
                logger.info(f"{_tmp_st_output}")

        # Freeze backbone
        if freeze_backbone:
            assert pretrained_backbone_path, "You shouldn't freeze a model without specifying pretrained_backbone_path"
            for _, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)
                self.backbone.eval()  # TODO: Check if needed??

        if self.difference_module == 'learnable':
            # Transformer Decoder
            self.decoder = ChangeNetClassifyDecoder(input_transform='multiple_select', in_index=in_index, align_corners=False,
                                                    in_channels=self.embed_dims, embedding_dim=self.embedding_dim, output_nc=output_nc,
                                                    feature_strides=feature_strides, model_name=self.model_name,
                                                    num_input=num_input, output_shape=output_shape, embed_dec=embed_dec,
                                                    learnable_difference_modules=learnable_difference_modules)

        elif self.difference_module == 'euclidean':
            self.dim_output = output_shape[0] // feature_strides[-1]
            self.dim_output1 = (output_shape[1] * num_input) // feature_strides[-1]
            self.fc_ip_dim = self.embed_dims[-1] * self.dim_output * self.dim_output1
            if 'vit' in self.model_name:
                self.fc_ip_dim = self.embed_dims[-1]

            self.embedding = embedding_vectors
            self.fc1 = nn.Sequential(
                nn.Linear(self.fc_ip_dim, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Linear(512, self.embedding))

        else:
            raise NotImplementedError('Difference module [%s] is not supported' % self.difference_module)

    def forward_once(self, x):
        """
        Forward pass of the Visual ChangeNetClassify model for a single image when using Euclidean Distance metric

        Args:
            x (torch.Tensor): Input tensor for the image input.

        Returns:
            torch.Tensor: Embedding vector for the given input image
        """
        if 'fan' in self.model_name:
            output = self.backbone(x)[-1]
        elif 'vit' in self.model_name:
            output = self.backbone(x)
        output = output.reshape(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, x1, x2):
        """
        Forward pass of the ChangeNetClassify model.

        Args:
            x1 (torch.Tensor): Input tensor for the first image input.
            x2 (torch.Tensor): Input tensor for the second image input.

        Returns:
            torch.Tensor: Returns the output tensor (For Arch1: two embedding vectors for 2 input images, Arch2: single output vector of dimension Batch_size x num_classes)
        """
        if 'vit' in self.model_name:
            assert x1.shape[2] == x1.shape[3], f"{self.model_name} backbone only supports square input images. " \
                "Please make sure the input height and width are equal, use an N x N grid to combine multiple lighting conditions."
            if self.difference_module == 'learnable':
                assert x1.shape[2] % 32 == 0 and x1.shape[3] % 32 == 0, "Input image size must be a multiple of 32 for ViT-Adapter"
        if self.difference_module == 'euclidean':
            # Classifier after last FM with eucledian distance:
            output1 = self.forward_once(x1)
            output2 = self.forward_once(x2)
            output = F.pairwise_distance(output1, output2)

        elif self.difference_module == 'learnable':
            [fx1, fx2] = [self.backbone(x1), self.backbone(x2)]
            out_decoder = self.decoder(fx1, fx2)
            output = out_decoder
        else:
            raise NotImplementedError('Only option 1 and 2 are supported')
        return output


def build_model(experiment_config,
                export=False):
    """ Build Visual ChangeNet classification model according to configuration

    Args:
        experiment_config: experiment configuration
        export: flag to indicate onnx export

    Returns:
        model

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset.classify

    backbone = model_config.backbone['type']
    freeze_backbone = model_config.backbone['freeze_backbone']
    feat_downsample = model_config.backbone.feat_downsample
    pretrained_backbone_path = model_config.backbone.pretrained_backbone_path

    channels_map = {"fan_tiny_8_p4_hybrid": [128, 256, 192, 192],
                    "fan_large_16_p4_hybrid": [128, 256, 480, 480],
                    "fan_small_12_p4_hybrid": [128, 256, 384, 384],
                    "fan_base_16_p4_hybrid": [128, 256, 448, 448],
                    "vit_large_nvdinov2": [1024, 1024, 1024, 1024]
                    }

    if backbone in channels_map:
        embed_dims = channels_map[backbone]
        model_config.decode_head.in_channels = embed_dims
    else:
        raise NotImplementedError('Backbone name [%s] is not supported' % backbone)

    embed_dim = model_config.decode_head.decoder_params['embed_dim']
    feature_strides = model_config.decode_head.feature_strides
    in_index = model_config.decode_head.in_index

    num_classes = dataset_config.num_classes
    image_width = dataset_config.image_width
    image_height = dataset_config.image_height
    num_input = dataset_config.num_input
    embedding_vectors = model_config.classify.embedding_vectors
    embed_dec = model_config.classify.embed_dec

    learnable_difference_modules = model_config.classify.learnable_difference_modules
    assert 1 <= learnable_difference_modules <= 4, "Visual ChangeNet only supports learnable difference modules in the range [1,4]"
    difference_module = model_config.classify.difference_module

    model = ChangeNetClassify(embed_dim=embed_dim,
                              model=backbone,
                              output_nc=num_classes,
                              input_nc=3,
                              embed_dims=embed_dims,
                              feature_strides=feature_strides,
                              in_index=in_index,
                              output_shape=[image_height, image_width],
                              num_input=num_input,
                              embedding_vectors=embedding_vectors,
                              embed_dec=embed_dec,
                              feat_downsample=feat_downsample,
                              learnable_difference_modules=learnable_difference_modules,
                              difference_module=difference_module,
                              pretrained_backbone_path=pretrained_backbone_path,
                              freeze_backbone=freeze_backbone)

    count_params(model)

    return model


def interpolate_vit_checkpoint(checkpoint, target_patch_size, target_resolution):
    """ Interpolate ViT backbone position embedding and patch embedding

    Args:
        checkpoint: pretrained ViT checkpoint
        target_patch_size: target patch size to interpolate to. ex: 14, 16, etc
        target_resolution: target image size to interpolate to. ex: 224, 512, 518, etc

    Returns:
        interpolated model checkpoints

    """
    if checkpoint is None:
        return checkpoint

    logger.info("Do ViT pretrained backbone interpolation")
    # interpolate patch embedding
    checkpoint = interpolate_patch_embed(checkpoint=checkpoint, new_patch_size=target_patch_size)

    # interpolate pos embedding
    checkpoint = interpolate_pos_embed(checkpoint_model=checkpoint,
                                       new_resolution=target_resolution,
                                       new_patch_size=target_patch_size)
    return checkpoint
