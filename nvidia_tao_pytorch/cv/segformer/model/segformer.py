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

"""SegFormer model builder"""

import numpy as np
import logging

import torch.nn as nn
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.cv.segformer.model.backbones import vit_adapter_model_dict, fan_model_dict, mit_model_dict, cradio_vit_adapter_model_dict
from nvidia_tao_pytorch.cv.segformer.model.decode_heads.segformer_head import TAOSegFormerHead
from nvidia_tao_pytorch.cv.segformer.model.segformer_utils import count_params
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import load_pretrained_weights

logger = logging.getLogger(__name__)


class SegFormer(nn.Module):
    """
    SegFormer model for semantic segmentation using a Transformer Encoder and Decoder.

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
        learnable_difference_modules (int): Number of decoder difference modules for Architecture 2
        pretrained_backbone_path (str): Path to the pre-trained backbone weights.
        activation_checkpoint (bool): Enable activation checkpointing
        freeze_backbone: Flag to freeze backbone weights during training.
        return_interm_indices (list): list of layer indices to reutrn as backbone features.
    """

    def __init__(self, input_nc=3, output_nc=2, embed_dim=256, model='fan_tiny_8_p4_hybrid_256', img_size=256,
                 in_channels=[64, 128, 320, 512], feature_strides=[4, 8, 16, 16], in_index=[0, 1, 2, 3], feat_downsample=False,
                 pretrained_backbone_path=None, return_interm_indices=[0, 1, 2, 3], activation_checkpoint=False, freeze_backbone=False):
        """Initialize SegFormer class"""
        super(SegFormer, self).__init__()

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
        self.in_channels = in_channels

        logger.info(f"Number of output classes: {output_nc}")
        assert img_size % feature_strides[-1] == 0, f"Input image size must be a multiple of {feature_strides[-1]}"

        # for fan backbone we load pretrained weights here, while for vit we load in the backbone
        if 'fan' in self.model_name:
            self.backbone = fan_model_dict[self.model_name](
                pretrained=False,
                num_classes=output_nc,
                checkpoint_path='',
                img_size=img_size,
                feat_downsample=feat_downsample
            )
            pretrained_backbone_ckp = load_pretrained_weights(pretrained_backbone_path) if pretrained_backbone_path else None
            if pretrained_backbone_ckp is not None:
                _tmp_st_output = self.backbone.load_state_dict(pretrained_backbone_ckp, strict=False)

                if get_global_rank() == 0:
                    logger.info(f"Loaded pretrained weights from {pretrained_backbone_path}")
                    logger.info(f"{_tmp_st_output}")

        elif 'mit' in self.model_name:
            self.backbone = mit_model_dict[self.model_name]()
            pretrained_backbone_ckp = load_pretrained_weights(pretrained_backbone_path) if pretrained_backbone_path else None
            if pretrained_backbone_ckp is not None:
                _tmp_st_output = self.backbone.load_state_dict(pretrained_backbone_ckp, strict=False)

                if get_global_rank() == 0:
                    logger.info(f"Loaded pretrained weights from {pretrained_backbone_path}")
                    logger.info(f"{_tmp_st_output}")

        elif 'radio' in self.model_name:
            assert img_size % 32 == 0, "Input image resolution must be a multiple of 32 for ViT-Adapter"
            self.backbone = cradio_vit_adapter_model_dict[self.model_name](
                out_indices=return_interm_indices,
                resolution=img_size,
                activation_checkpoint=activation_checkpoint,
                use_summary_token=True)
            pretrained_backbone_ckp = load_pretrained_weights(pretrained_backbone_path) if pretrained_backbone_path else None
            if pretrained_backbone_ckp is not None:
                _tmp_st_output = self.backbone.load_state_dict(pretrained_backbone_ckp, strict=False)

                if get_global_rank() == 0:
                    logger.info(f"Loaded pretrained weights from {pretrained_backbone_path}")
                    logger.info(f"{_tmp_st_output}")

        elif 'vit' in self.model_name:
            assert img_size % 32 == 0, "Input image resolution must be a multiple of 32 for ViT-Adapter"
            self.backbone = vit_adapter_model_dict[self.model_name](
                out_indices=return_interm_indices,
                resolution=img_size,
                init_cfg={
                    "checkpoint": pretrained_backbone_path
                },
                activation_checkpoint=activation_checkpoint)

        else:
            raise NotImplementedError('Bacbkbone name [%s] is not supported' % self.model_name)

        # Freeze backbone
        if freeze_backbone:
            assert pretrained_backbone_path is not None, "You shouldn't freeze a model without specifying pretrained_backbone_path"
            for _, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)
            # self.backbone.eval()  # TODO: Check if needed??
            logger.info("Frozen backbone training")

        # Transformer Decoder
        self.decoder = TAOSegFormerHead(
            input_transform='multiple_select', in_index=in_index, align_corners=False,
            in_channels=self.in_channels, embedding_dim=self.embedding_dim, output_nc=output_nc,
            feature_strides=feature_strides, model_name=self.model_name
        )

    def forward(self, x):
        """
        Forward pass of the SegFormer model.

        Args:
            x1 (torch.Tensor): Input tensor for the first image input.
            x2 (torch.Tensor): Input tensor for the second image input.

        Returns:
            torch.Tensor: Output tensor representing the segmentation map.
        """
        f = self.backbone(x)
        out_decoder = self.decoder(f)
        return out_decoder


def build_model(experiment_config,
                export=False):
    """ Build segformer model according to configuration

    Args:
        experiment_config: experiment configuration
        export: flag to indicate onnx export

    Returns:
        model

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset.segment

    backbone = model_config.backbone['type']
    freeze_backbone = model_config.backbone['freeze_backbone']
    feat_downsample = model_config.backbone.feat_downsample
    pretrained_backbone_path = model_config.backbone.pretrained_backbone_path

    # We need these because the multiple select feature from these backbone has fixed feature dimensions
    channels_map = {
        "mit_b0": [32, 64, 160, 256],
        "mit_b1": [64, 128, 320, 512],
        "mit_b2": [64, 128, 320, 512],
        "mit_b3": [64, 128, 320, 512],
        "mit_b4": [64, 128, 320, 512],
        "mit_b5": [64, 128, 320, 512],
        "fan_tiny_8_p4_hybrid": [128, 256, 192, 192],
        "fan_large_16_p4_hybrid": [128, 256, 480, 480],
        "fan_small_12_p4_hybrid": [128, 256, 384, 384],
        "fan_base_16_p4_hybrid": [128, 256, 448, 448],
        "vit_large_nvdinov2": [1024, 1024, 1024, 1024],
        "vit_giant_nvdinov2": [1536, 1536, 1536, 1536],
        "vit_base_nvclip_16_siglip": [768, 768, 768, 768],
        "vit_huge_nvclip_14_siglip": [1280, 1280, 1280, 1280],
        "c_radio_v2_vit_base_patch16_224": [768, 768, 768, 768],
        "c_radio_v2_vit_large_patch16_224": [1024, 1024, 1024, 1024],
        "c_radio_v2_vit_huge_patch16_224": [1280, 1280, 1280, 1280]
    }

    if backbone in channels_map:
        in_channels = channels_map[backbone]
        model_config.decode_head.in_channels = in_channels
    else:
        raise NotImplementedError('Bacbkbone name [%s] is not supported' % backbone)

    embed_dim = model_config.decode_head.decoder_params['embed_dim']
    feature_strides = model_config.decode_head.feature_strides
    in_channels = model_config.decode_head.in_channels
    in_index = model_config.decode_head.in_index

    num_classes = dataset_config.num_classes
    img_size = dataset_config.img_size

    model = SegFormer(
        input_nc=3,
        output_nc=num_classes,
        embed_dim=embed_dim,
        model=backbone,
        img_size=img_size,
        in_channels=in_channels,
        feature_strides=feature_strides,
        in_index=in_index,
        feat_downsample=feat_downsample,
        pretrained_backbone_path=pretrained_backbone_path,
        freeze_backbone=freeze_backbone
    )
    count_params(model)

    return model
