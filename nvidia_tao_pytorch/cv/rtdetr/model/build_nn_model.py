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

"""The build nn module model."""

import torch.nn as nn

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logging

from nvidia_tao_pytorch.cv.rtdetr.model.backbone.resnet import resnet_model_dict
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.convnext import convnext_model_dict
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.fan import fan_model_dict
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.efficientvit import efficientvit_model_dict

from nvidia_tao_pytorch.cv.rtdetr.model.hybrid_encoder import HybridEncoder
from nvidia_tao_pytorch.cv.rtdetr.model.rtdetr_decoder import RTDETRTransformer
from nvidia_tao_pytorch.cv.rtdetr.model.rtdetr import RTDETR
from nvidia_tao_pytorch.cv.rtdetr.utils.misc import load_pretrained_weights


class RTDETRModel(nn.Module):
    """RT-DETR model module."""

    def __init__(self,
                 backbone_name='resnet_50',
                 pretrained_backbone=None,
                 train_backbone=True,
                 num_classes=80,
                 out_indices=[1, 2, 3],
                 # Encoder
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0,
                 enc_act='gelu',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1,
                 act='silu',
                 eval_spatial_size=[640, 640],
                 # Decoder
                 feat_channels=[256, 256, 256],
                 num_levels=3,
                 num_queries=300,
                 num_decoder_layers=6,
                 num_denoising=100,
                 eval_idx=-1,
                 multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800],
                 activation_checkpoint=False,
                 frozen_fm_cfg=None,
                 export=False,
                 ):
        """Initialize RT-DETR Model."""
        super().__init__()
        parser = None
        if backbone_name.startswith('resnet'):
            backbone = resnet_model_dict[backbone_name](
                out_indices,
            )
            for name, parameter in backbone.named_parameters():
                if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                    parameter.requires_grad_(False)
            in_channels = backbone.out_channels
        elif backbone_name.startswith('convnext'):
            backbone = convnext_model_dict[backbone_name](
                out_indices
            )
            for name, parameter in backbone.named_parameters():
                if not train_backbone:
                    parameter.requires_grad_(False)
            in_channels = backbone.out_channels

            def parse_convnext_ptm(m):
                """Parse official checkpoint from Meta"""
                if "model" in m:
                    return m["model"]
                return m
            parser = parse_convnext_ptm
        elif backbone_name.startswith('fan'):
            backbone = fan_model_dict[backbone_name](
                out_indices, activation_checkpoint=activation_checkpoint,
            )
            for name, parameter in backbone.named_parameters():
                if not train_backbone:
                    parameter.requires_grad_(False)
            in_channels = [o for i, o in enumerate(backbone.out_channels) if i in out_indices]
        elif backbone_name.startswith('efficientvit'):
            backbone = efficientvit_model_dict[backbone_name](
                out_indices
            )
            for name, parameter in backbone.named_parameters():
                if not train_backbone:
                    parameter.requires_grad_(False)
            in_channels = backbone.out_channels
        else:
            raise NotImplementedError(f"{backbone_name} is not supported")

        pretrained_backbone_ckp = load_pretrained_weights(pretrained_backbone, parser=parser) if pretrained_backbone else None
        if pretrained_backbone_ckp:
            _tmp_st_output = backbone.load_state_dict(pretrained_backbone_ckp, strict=False)
            if get_global_rank() == 0:
                logging.info(f"Loaded pretrained weights from {pretrained_backbone}")
                logging.info(f"{_tmp_st_output}")

        encoder = HybridEncoder(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            frozen_fm_cfg=frozen_fm_cfg,
        )

        decoder = RTDETRTransformer(
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            num_levels=num_levels,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_denoising=num_denoising,
            eval_idx=eval_idx,
            eval_spatial_size=eval_spatial_size,
            num_classes=num_classes,
            frozen_fm_cfg=frozen_fm_cfg,
            export=export,
        )

        self.model = RTDETR(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            multi_scale=multi_scale,
            frozen_fm_cfg=frozen_fm_cfg,
            export=export,
        )

    def forward(self, x, targets=None):
        """model forward function"""
        x = self.model(x, targets)
        return x


def build_model(experiment_config,
                export=False):
    """ Build dino model according to configuration.

    Args:
        experiment_config (OmegaConf): experiment configuration.
        export (bool): flag to indicate onnx export.

    Returns:
        model (nn.Module): RT-DETR model.
    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset

    backbone = model_config.backbone
    train_backbone = model_config.train_backbone

    pretrained_backbone = model_config.pretrained_backbone_path
    return_interm_indices = model_config.return_interm_indices

    eval_spatial_size = dataset_config.augmentation.eval_spatial_size
    multi_scale = dataset_config.augmentation.multi_scales
    num_classes = dataset_config.num_classes
    num_queries = model_config.num_queries
    hidden_dim = model_config.hidden_dim
    use_encoder_idx = model_config.use_encoder_idx
    nhead = model_config.nheads
    dim_feedforward = model_config.dim_feedforward
    dropout = model_config.dropout_ratio
    pe_temperature = model_config.pe_temperature
    expansion = model_config.expansion
    depth_mult = model_config.depth_mult
    enc_act = model_config.enc_act
    act = model_config.act
    num_levels = model_config.num_feature_levels
    num_encoder_layers = model_config.enc_layers
    num_decoder_layers = model_config.dec_layers
    num_denoising = model_config.dn_number
    feat_channels = model_config.feat_channels
    feat_strides = model_config.feat_strides
    eval_idx = model_config.eval_idx
    frozen_fm_cfg = model_config.frozen_fm

    activation_checkpoint = experiment_config.train.activation_checkpoint

    model = RTDETRModel(
        backbone_name=backbone,
        train_backbone=train_backbone,
        pretrained_backbone=pretrained_backbone,
        out_indices=return_interm_indices,
        num_classes=num_classes,
        eval_spatial_size=eval_spatial_size,
        multi_scale=multi_scale,
        num_queries=num_queries,

        # Encoder
        hidden_dim=hidden_dim,
        use_encoder_idx=use_encoder_idx,
        num_encoder_layers=num_encoder_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        enc_act=enc_act,
        pe_temperature=pe_temperature,
        expansion=expansion,
        depth_mult=depth_mult,
        act=act,
        # Decoder
        feat_channels=feat_channels,
        feat_strides=feat_strides,
        num_levels=num_levels,
        num_decoder_layers=num_decoder_layers,
        num_denoising=num_denoising,
        eval_idx=eval_idx,
        activation_checkpoint=activation_checkpoint,
        # frozen FM
        frozen_fm_cfg=frozen_fm_cfg,
        export=export,
    )
    return model
