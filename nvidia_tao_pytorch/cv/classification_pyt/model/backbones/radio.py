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


""" RADIO Model Module """

from nvidia_tao_pytorch.cv.backbone.radio import RADIOWrapper, CRADIO
from nvidia_tao_pytorch.cv.backbone.radio.utils import get_prefix_state_dict
from nvidia_tao_pytorch.cv.backbone.radio.model_cfg import radio_model_cfg
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logging

import torch
from torch import nn


class CRadioP1VitHugePatch16Mlpnorm(nn.Module):
    """
    CRADIO p1 ViT Huge model
    """

    def __init__(self, *args, freeze=False, init_cfg=None, resolution, **kwargs):
        """CRADIO p1 ViT Huge model

        Args:
            resolution (tuple): input resolution
            freeze (bool, optional): whether to freeze backbone. Defaults to False.
            init_cfg (dict, optional): config. Defaults to None.
        """
        super().__init__()

        model_cfg = radio_model_cfg["c_radio_p1_vit_huge_patch16_224_mlpnorm"]
        backbone = CRADIO(backbone="vit_huge_patch16_224_mlpnorm",
                          **model_cfg,
                          **kwargs)

        self.freeze = freeze
        pretrained = None
        if init_cfg and init_cfg.get("checkpoint"):
            pretrained = init_cfg["checkpoint"]
            backbone.load_state_dict(checkpoint=pretrained)
            if get_global_rank() == 0:
                logging.info(f"Loaded pretrained weights from {pretrained}")

        if self.freeze:
            assert pretrained is not None, "You shouldn't freeze a model without specifying pretrained"
            backbone.eval()

            for p in backbone.parameters():
                p.requires_grad = False

        self.radio = RADIOWrapper(model=backbone, resolution=resolution)

    def forward(self, x):
        """
        Forward function and return the features
        """
        summary, _ = self.radio(x)

        return summary


class CRadioP2VitHugePatch16Mlpnorm(nn.Module):
    """
    CRADIO p2 ViT Huge model
    """

    def __init__(self, *args, freeze=False, init_cfg=None, resolution, **kwargs):
        """CRADIO p2 ViT Huge model

        Args:
            resolution (tuple): input resolution
            freeze (bool, optional): whether to freeze backbone. Defaults to False.
            init_cfg (dict, optional): config. Defaults to None.
        """
        super().__init__()

        model_cfg = radio_model_cfg["c_radio_p2_vit_huge_patch16_224_mlpnorm"]
        backbone = CRADIO(backbone="vit_huge_patch16_224_mlpnorm",
                          **model_cfg,
                          **kwargs)

        self.freeze = freeze
        pretrained = None
        if init_cfg and init_cfg.get("checkpoint"):
            pretrained = init_cfg["checkpoint"]
            backbone.load_state_dict(checkpoint=pretrained)
            if get_global_rank() == 0:
                logging.info(f"Loaded pretrained weights from {pretrained}")

        if self.freeze:
            assert pretrained is not None, "You shouldn't freeze a model without specifying pretrained"
            backbone.eval()

            for p in backbone.parameters():
                p.requires_grad = False

        self.radio = RADIOWrapper(model=backbone, resolution=resolution)

    def forward(self, x):
        """
        Forward function and return the features
        """
        summary, _ = self.radio(x)

        return summary


class CRadioP3VitHugePatch16Mlpnorm(nn.Module):
    """
    CRADIO p3 ViT Huge model
    """

    def __init__(self, *args, freeze=False, init_cfg=None, resolution, **kwargs):
        """CRADIO p1 ViT Huge model

        Args:
            resolution (tuple): input resolution
            freeze (bool, optional): whether to freeze backbone. Defaults to False.
            init_cfg (dict, optional): config. Defaults to None.
        """
        super().__init__()

        model_cfg = radio_model_cfg["c_radio_p3_vit_huge_patch16_224_mlpnorm"]
        backbone = CRADIO(backbone="vit_huge_patch16_224_mlpnorm",
                          **model_cfg,
                          **kwargs)

        self.freeze = freeze
        pretrained = None
        if init_cfg and init_cfg.get("checkpoint"):
            pretrained = init_cfg["checkpoint"]
            backbone.load_state_dict(checkpoint=pretrained)
            if get_global_rank() == 0:
                logging.info(f"Loaded pretrained weights from {pretrained}")

        if self.freeze:
            assert pretrained is not None, "You shouldn't freeze a model without specifying pretrained"
            backbone.eval()

            for p in backbone.parameters():
                p.requires_grad = False

        self.radio = RADIOWrapper(model=backbone, resolution=resolution)

    def forward(self, x):
        """
        Forward function and return the features
        """
        summary, _ = self.radio(x)

        return summary


class CRadioV2VitHugePatch16(nn.Module):
    """
    CRADIO v2 ViT Huge model
    """

    def __init__(self, *args, freeze=False, init_cfg=None, resolution, **kwargs):
        """CRADIO v2 ViT Huge model

        Args:
            resolution (tuple): input resolution
            freeze (bool, optional): whether to freeze backbone. Defaults to False.
            init_cfg (dict, optional): config. Defaults to None.
        """
        super().__init__()

        model_cfg = radio_model_cfg["c_radio_v2_vit_huge_patch16_224"]
        backbone = CRADIO(backbone="vit_huge_patch16_224",
                          **model_cfg,
                          **kwargs)

        self.freeze = freeze
        pretrained = None
        if init_cfg and init_cfg.get("checkpoint"):
            pretrained = init_cfg["checkpoint"]
            checkpoint = torch.load(pretrained, map_location="cpu")
            key_warn = backbone.model.load_state_dict(get_prefix_state_dict(checkpoint, "radio_model.model."))
            if key_warn.missing_keys:
                print(f"Missing keys in state dict: {key_warn.missing_keys}")
            if key_warn.unexpected_keys:
                print(f"Unexpected keys in state dict: {key_warn.unexpected_keys}")
            print(f"Loaded pretrained weights from {pretrained}")

        if self.freeze:
            assert pretrained is not None, "You shouldn't freeze a model without specifying pretrained"
            backbone.eval()

            for p in backbone.parameters():
                p.requires_grad = False

        self.radio = RADIOWrapper(model=backbone, resolution=resolution)

    def forward(self, x):
        """
        Forward function and return the features
        """
        summary, _ = self.radio(x)

        return summary


class CRadioV2VitLargePatch16(nn.Module):
    """
    CRADIO v2 ViT large model
    """

    def __init__(self, *args, freeze=False, init_cfg=None, resolution, **kwargs):
        """CRADIO v2 ViT large model

        Args:
            resolution (tuple): input resolution
            freeze (bool, optional): whether to freeze backbone. Defaults to False.
            init_cfg (dict, optional): config. Defaults to None.
        """
        super().__init__()

        model_cfg = radio_model_cfg["c_radio_v2_vit_large_patch16_224"]
        backbone = CRADIO(backbone="vit_large_patch16_224",
                          **model_cfg,
                          **kwargs)

        self.freeze = freeze
        pretrained = None
        if init_cfg and init_cfg.get("checkpoint"):
            pretrained = init_cfg["checkpoint"]
            checkpoint = torch.load(pretrained, map_location="cpu")
            key_warn = backbone.model.load_state_dict(get_prefix_state_dict(checkpoint, "radio_model.model."))
            if key_warn.missing_keys:
                print(f"Missing keys in state dict: {key_warn.missing_keys}")
            if key_warn.unexpected_keys:
                print(f"Unexpected keys in state dict: {key_warn.unexpected_keys}")
            print(f"Loaded pretrained weights from {pretrained}")

        if self.freeze:
            assert pretrained is not None, "You shouldn't freeze a model without specifying pretrained"
            backbone.eval()

            for p in backbone.parameters():
                p.requires_grad = False

        self.radio = RADIOWrapper(model=backbone, resolution=resolution)

    def forward(self, x):
        """
        Forward function and return the features
        """
        summary, _ = self.radio(x)

        return summary


class CRadioV2VitBasePatch16(nn.Module):
    """
    CRADIO v2 ViT base model
    """

    def __init__(self, *args, freeze=False, init_cfg=None, resolution, **kwargs):
        """CRADIO v2 ViT Huge model

        Args:
            resolution (tuple): input resolution
            freeze (bool, optional): whether to freeze backbone. Defaults to False.
            init_cfg (dict, optional): config. Defaults to None.
        """
        super().__init__()

        model_cfg = radio_model_cfg["c_radio_v2_vit_base_patch16_224"]
        backbone = CRADIO(backbone="vit_base_patch16_224",
                          **model_cfg,
                          **kwargs)

        self.freeze = freeze
        pretrained = None
        if init_cfg and init_cfg.get("checkpoint"):
            pretrained = init_cfg["checkpoint"]
            checkpoint = torch.load(pretrained, map_location="cpu")
            key_warn = backbone.model.load_state_dict(get_prefix_state_dict(checkpoint, "radio_model.model."))
            if key_warn.missing_keys:
                print(f"Missing keys in state dict: {key_warn.missing_keys}")
            if key_warn.unexpected_keys:
                print(f"Unexpected keys in state dict: {key_warn.unexpected_keys}")
            print(f"Loaded pretrained weights from {pretrained}")

        if self.freeze:
            assert pretrained is not None, "You shouldn't freeze a model without specifying pretrained"
            backbone.eval()

            for p in backbone.parameters():
                p.requires_grad = False

        self.radio = RADIOWrapper(model=backbone, resolution=resolution)

    def forward(self, x):
        """
        Forward function and return the features
        """
        summary, _ = self.radio(x)

        return summary
