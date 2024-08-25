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

""" DINOv2 ViT Model Module """

import math
from functools import partial
import copy
import logging
from logging import FileHandler
from collections import defaultdict

import torch
import torch.nn as nn

from timm.layers import PatchEmbed, SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer

from mmpretrain.registry import MODELS
from mmengine.dist import master_only
from mmengine.logging import MMLogger, print_log
from mmengine.model.weight_init import PretrainedInit, initialize, update_init_info
from mmengine.model.wrappers.utils import is_model_wrapper


def interpolate_pos_encoding(pos_embed, w, h, patch_size=14):
    """
    Interpolate the position encodings for the model to accept any image size.
    """
    w0 = w // patch_size
    h0 = h // patch_size
    npatch = w0 * h0
    N = pos_embed.shape[1] - 1

    if npatch == N and w == h:
        return pos_embed

    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8

    w0, h0 = w0 + 0.1, h0 + 0.1

    # We need fp32 for the interpolation
    reshaped_pos_embed = patch_pos_embed.reshape(
        1, int(math.sqrt(N)), int(math.sqrt(N)), pos_embed.shape[-1]
    ).permute(0, 3, 1, 2)

    patch_pos_embed = nn.functional.interpolate(
        reshaped_pos_embed,
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode="bicubic",
    )

    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1], "The interpolated value does not match the positional embedding size."
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(
        1, -1, pos_embed.shape[-1]
    )

    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


class DinoV2ViT(VisionTransformer):
    """
    This class extends the VisionTransformer class from timm library so that we can
    handle different image sizes.
    """

    def __init__(self, *args, **kwargs):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""
        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.
        init_cfg = kwargs.pop('init_cfg', None)
        self.init_cfg = None
        self._is_init = False
        if init_cfg is not None:
            self.init_cfg = copy.deepcopy(init_cfg)
        register_tokens = kwargs.pop('register_tokens', 0)
        self.num_register_tokens = register_tokens
        super(DinoV2ViT, self).__init__(*args, **kwargs)

        if register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, register_tokens, self.embed_dim)
            )

    @property
    def is_init(self):
        """Return is_init."""
        return self._is_init

    @is_init.setter
    def is_init(self, value):
        self._is_init = value

    def init_weights(self, weight_init=None):
        """Initialize the weights."""
        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for _, param in self.named_parameters():
                self._params_init_info[param][
                    'init_info'] = f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self._params_init_info[param][
                    'tmp_mean_value'] = param.data.mean().cpu()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}',
                    logger='current',
                    level=logging.DEBUG)

                init_cfgs = self.init_cfg
                if isinstance(self.init_cfg, dict):
                    init_cfgs = [self.init_cfg]

                # PretrainedInit has higher priority than any other init_cfg.
                # Therefore we initialize `pretrained_cfg` last to overwrite
                # the previous initialized weights.
                # See details in https://github.com/open-mmlab/mmengine/issues/691 # noqa E501
                other_cfgs = []
                pretrained_cfg = []
                for init_cfg in init_cfgs:
                    assert isinstance(init_cfg, dict)
                    if (init_cfg['type'] == 'Pretrained' or init_cfg['type'] is PretrainedInit):
                        pretrained_cfg.append(init_cfg)
                    else:
                        other_cfgs.append(init_cfg)

                initialize(self, other_cfgs)

            for m in self.children():
                if is_model_wrapper(m) and not hasattr(m, 'init_weights'):
                    m = m.module
                if hasattr(m, 'init_weights') and not getattr(
                        m, 'is_init', False):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f'Initialized by '
                        f'user-defined `init_weights`'
                        f' in {m.__class__.__name__} ')
            if self.init_cfg and pretrained_cfg:
                initialize(self, pretrained_cfg)
            self._is_init = True
        else:
            print_log(
                f'init_weights of {self.__class__.__name__} has '
                f'been called more than once.',
                logger='current',
                level=logging.WARNING)

        if is_top_level_module:
            self._dump_init_info()

            for sub_module in self.modules():
                del sub_module._params_init_info

    @master_only
    def _dump_init_info(self):
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.
        """
        logger = MMLogger.get_current_instance()
        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write(
                    'Name of parameter - Initialization information\n')
                for name, param in self.named_parameters():
                    handler.stream.write(
                        f'\n{name} - {param.shape}: '
                        f"\n{self._params_init_info[param]['init_info']} \n")
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                logger.info(
                    f'\n{name} - {param.shape}: '
                    f"\n{self._params_init_info[param]['init_info']} \n ")

    def __interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8

        w0, h0 = w0 + 0.1, h0 + 0.1

        # We need fp32 for the interpolation
        reshaped_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(N)), int(math.sqrt(N)), dim
        ).permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            reshaped_pos_embed,
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1], "The interpolated value does not match the positional embedding size."
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def _pos_embed(self, x):
        B, S, _ = x.shape
        w = h = int(math.sqrt(S)) * self.patch_embed.patch_size[0]

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.__interpolate_pos_encoding(x, w, h)

        # add register tokens
        if self.num_register_tokens > 0:
            x = torch.cat((x, self.register_tokens.expand(B, -1, -1)), dim=1)

        return self.pos_drop(x)


@MODELS.register_module()
class vit_large_patch14_dinov2_swiglu(DinoV2ViT):
    """
    DINOV2 ViT Large model with SwiGLU activation
    """

    def __init__(self, *args, freeze=False, pretrained=None, init_cfg=None, **kwargs):
        """Initialize"""
        model_kwargs = dict(
            patch_size=14,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            init_values=1e-5,
            img_size=518,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
            mlp_ratio=5472 / 1024,
            embed_layer=partial(PatchEmbed, strict_img_size=False),
            global_pool="token",
            num_classes=0,
            init_cfg=init_cfg,
            **kwargs
        )
        super(vit_large_patch14_dinov2_swiglu, self).__init__(*args, **model_kwargs)

        self.freeze = freeze
        if init_cfg:
            pretrained = init_cfg["checkpoint"]
            model = torch.load(pretrained, "cpu")
            self.load_state_dict(model, strict=True)
            print(f"Loaded pretrained weights from {pretrained}")

        if freeze:
            assert pretrained is not None, "You shouldn't freeze a model without specifying pretrained"
            self.eval()

            for p in self.parameters():
                p.requires_grad = False

    def forward(self, *args, **kwargs):
        """
        Forward function and return the flatten output (cls_token)
        """
        if self.freeze:
            self.eval()

        x = super().forward(*args, **kwargs)

        return x.flatten(1)


@MODELS.register_module()
class vit_giant_patch14_reg4_dinov2_swiglu(DinoV2ViT):
    """
    DINOV2 ViT Giant model with SwiGLU activation
    """

    def __init__(self, *args, freeze=False, pretrained=None, init_cfg=None, **kwargs):
        """Initialize"""
        model_kwargs = dict(
            patch_size=14,
            embed_dim=1536,
            depth=40,
            num_heads=24,
            init_values=1e-5,
            img_size=518,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
            mlp_ratio=8192 / 1536,
            embed_layer=partial(PatchEmbed, strict_img_size=False),
            global_pool="token",
            num_classes=0,
            register_tokens=4,
            init_cfg=init_cfg,
            **kwargs
        )
        super(vit_giant_patch14_reg4_dinov2_swiglu, self).__init__(*args, **model_kwargs)

        self.freeze = freeze
        if init_cfg:
            pretrained = init_cfg["checkpoint"]
            model = torch.load(pretrained, "cpu")
            self.load_state_dict(model, strict=True)
            print(f"Loaded pretrained weights from {pretrained}")

        if freeze:
            assert pretrained is not None, "You shouldn't freeze a model without specifying pretrained"
            self.eval()

            for p in self.parameters():
                p.requires_grad = False

    def forward(self, *args, **kwargs):
        """
        Forward function and return the flatten output (cls_token)
        """
        if self.freeze:
            self.eval()

        x = super().forward(*args, **kwargs)

        return x.flatten(1)
