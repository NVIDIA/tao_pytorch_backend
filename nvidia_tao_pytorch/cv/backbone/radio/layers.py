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

"""Layer components for RADIO model
"""

import math
from typing import Union, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


class ClsToken(nn.Module):
    """Class token class for RADIO ViT backbone
    """

    def __init__(self, ndim: int,
                 num_tokens: int = 1,
                 enabled: bool = True,
                 register_multiple: int = 0
                 ):
        """Class token class for RADIO ViT backbone

        Args:
            ndim (int): class token dimension
            num_tokens (int, optional): number of class tokens. Defaults to 1.
            enabled (bool, optional): whether to enable class token. Defaults to True.
            register_multiple (int, optional): number of extra tokens. Defaults to 0.
        """
        super().__init__()

        self.ndim = ndim
        self.enabled = enabled
        self.num_registers = 0
        self.num_tokens = num_tokens
        if enabled:
            if register_multiple > 0:
                self.num_registers = register_multiple - (num_tokens % register_multiple)

            scale = ndim ** -0.5
            self.token = nn.Parameter(torch.randn(num_tokens + self.num_registers, ndim) * scale)
        else:
            self.token = None

        self.num_patches = self.num_tokens + self.num_registers

    def disable(self):
        """Method to disable everything
        """
        self.token = None
        self.enabled = False

    def forward(self, x: torch.Tensor):
        """Forward function and return the input with class tokens concatenated
        """
        if self.token is None:
            return x

        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([
            token,
            x,
        ], dim=1)

        return x

    def no_weight_decay(self):
        """No weight decay
        """
        return [
            'token',
        ]


class ViTPatchGenerator(nn.Module):
    """ViT patch generator class for CPE (Cropped Position Embedding) purpose
    """

    def __init__(self,
                 patch_size: int,
                 embed_dim: int,
                 input_dims: Union[int, Tuple[int, int]],
                 abs_pos: bool = True,
                 normalize_patches: bool = False,
                 cls_token: bool = False,
                 max_input_dims: Optional[Union[int, Tuple[int, int]]] = None,
                 pos_dropout: float = 0.0,
                 return_pos_enc: bool = False,
                 num_cls_tokens: int = 1,
                 register_multiple: int = 0,
                 device=None, dtype=None
                 ):
        """ViT patch generator class for CPE (Cropped Position Embedding) purpose

        Args:
            patch_size (int): patch size
            embed_dim (int): patch embedding dimension
            input_dims (Union[int, Tuple[int, int]]): input dimension
            abs_pos (bool, optional): absolute position embedding. Defaults to True.
            normalize_patches (bool, optional): whether to normalize patch. Defaults to False.
            cls_token (bool, optional): whether to enable class token. Defaults to False.
            max_input_dims (Optional[Union[int, Tuple[int, int]]], optional): maximum input dimension. Defaults to None.
            pos_dropout (float, optional): position embedding dropout rate. Defaults to 0.0.
            return_pos_enc (bool, optional): whether to return position embedding. Defaults to False.
            num_cls_tokens (int, optional): number of class tokens. Defaults to 1.
            register_multiple (int, optional): number of extra tokens. Defaults to 0.
            device (str, optional): device. Defaults to None.
            dtype (str, optional): data type. Defaults to None.
        """
        super().__init__()

        if isinstance(input_dims, int):
            input_dims = (input_dims, input_dims)

        if max_input_dims is None:
            max_input_dims = input_dims
        if isinstance(max_input_dims, int):
            max_input_dims = (max_input_dims, max_input_dims)

        max_input_dims = tuple(
            int(math.ceil(d / patch_size) * patch_size)
            for d in max_input_dims
        )

        self.cpe_mode = max_input_dims != input_dims
        self.pos_dropout = pos_dropout
        self.return_pos_enc = return_pos_enc

        factory = dict(device=device, dtype=dtype)

        self.patch_size = patch_size
        self.abs_pos = abs_pos
        self.embed_dim = embed_dim

        self.num_rows = max_input_dims[0] // patch_size
        self.num_cols = max_input_dims[1] // patch_size
        self.input_dims = tuple(d // patch_size for d in input_dims)
        self.num_patches = self.num_rows * self.num_cols
        self.max_input_dims = max_input_dims

        self.im_to_patches = Im2Patches(patch_size)
        self.embedder = ViTPatchLinear(patch_size, embed_dim, **factory)

        if abs_pos:
            scale = embed_dim ** -0.5
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim, **factory) * scale)

        self.cls_token = ClsToken(
            embed_dim,
            num_tokens=num_cls_tokens,
            enabled=cls_token,
            register_multiple=register_multiple,
        )

        self.patch_normalizer = nn.LayerNorm(embed_dim) if normalize_patches else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function to return the patch embeddings with position embedding applied
        """
        patches = self.embed_patches(x)
        patches, pos_enc = self.apply_pos_enc(patches, input_size=x.shape[2:])
        patches = self.cls_token(patches)
        patches = self.patch_normalizer(patches)
        if self.return_pos_enc:
            return patches, pos_enc

        return patches

    @property
    def apply_cls_token(self):
        """Check whether class token is applied
        """
        return self.cls_token.enabled

    @property
    def num_cls_tokens(self):
        """Number of class tokens
        """
        return self.cls_token.num_tokens

    @property
    def num_registers(self):
        """Number of register tokens
        """
        return self.cls_token.num_registers

    @property
    def num_skip(self):
        """Total number of extra tokens (class tokens + register tokens)
        """
        return self.num_cls_tokens + self.num_registers

    def no_weight_decay(self):
        """No weight decay
        """
        return [
            'pos_embed',
        ]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if self.abs_pos:
            self._load_embed(state_dict[f'{prefix}pos_embed'], self.pos_embed)

    def _load_embed(self, src_embed: torch.Tensor, targ_embed: nn.Parameter):
        if src_embed.shape != targ_embed.shape:
            src_size = int(math.sqrt(src_embed.shape[1]))

            assert src_size ** 2 == src_embed.shape[1], 'Unable to interpolate non-square embedding'

            src_embed = rearrange(src_embed, 'b (h w) c -> b c h w', h=src_size, w=src_size)
            src_embed = F.interpolate(src_embed, size=(self.num_rows, self.num_cols), mode='bicubic', align_corners=True, antialias=False)
            src_embed = rearrange(src_embed, 'b c h w -> b (h w) c')
        targ_embed.data.copy_(src_embed)

    def _load_projection(self, src_proj_weight: torch.Tensor, targ_proj_weight: torch.Tensor):
        if src_proj_weight.shape != targ_proj_weight.shape:
            src_patch_size = int(math.sqrt(src_proj_weight.shape[1] // 3))

            assert (src_patch_size ** 2) * 3 == src_proj_weight.shape[1], 'Unable to interpolate non-square patch size'

            src_proj_weight = rearrange(src_proj_weight, 'b (c h w) -> b c h w', c=3, h=src_patch_size, w=src_patch_size)
            src_proj_weight = F.interpolate(src_proj_weight, size=(self.patch_size, self.patch_size), mode='bicubic', align_corners=True, antialias=False)
            src_proj_weight = rearrange(src_proj_weight, 'b c h w -> b (c h w)')
        targ_proj_weight.data.copy_(src_proj_weight)

    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Embed image patches
        """
        patches = self.im_to_patches(x)
        patches = self.embedder(patches)

        return patches

    def apply_pos_enc(self,
                      patches: torch.Tensor,
                      patch_idxs: Optional[torch.Tensor] = None,
                      input_size: Optional[Tuple[int, int]] = None
                      ) -> torch.Tensor:
        """Apply position embedding
        """
        if not self.abs_pos:
            return patches

        pos_enc = self.get_pos_enc(patches.shape[0], patch_idxs, input_size)

        if self.training and self.pos_dropout > 0:
            keeps = torch.rand(patches.shape[0], 1, 1, dtype=pos_enc.dtype, device=pos_enc.device) > self.pos_dropout
            pos_enc_drop = torch.where(keeps, pos_enc, 0)
        else:
            pos_enc_drop = pos_enc

        return patches + pos_enc_drop, pos_enc

    def get_pos_enc(self,
                    batch_size: int,
                    patch_idxs: Optional[torch.Tensor] = None,
                    input_size: Optional[Tuple[int, int]] = None
                    ) -> torch.Tensor:
        """Get position embedding
        """
        if input_size is None:
            input_dims = self.input_dims
        else:
            input_dims = tuple(d // self.patch_size for d in input_size)

        pos_embed = self._get_pos_embeddings(batch_size, input_dims)

        if patch_idxs is None:
            return pos_embed

        exp_patch_idxs = patch_idxs.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1])

        pos_embed = torch.gather(pos_embed.expand(patch_idxs.shape[0], -1, -1), dim=1, index=exp_patch_idxs)
        return pos_embed

    def _get_pos_embeddings(self, batch_size: int, input_dims: Tuple[int, int]):
        if (self.num_rows, self.num_cols) == input_dims:
            return self.pos_embed

        pos_embed = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1).permute(0, 3, 1, 2)

        def window_select(pos_embed):
            if input_dims[0] < pos_embed.shape[-2]:
                pos_embed = pos_embed[..., :input_dims[0], :]
            if input_dims[1] < pos_embed.shape[-1]:
                pos_embed = pos_embed[..., :, :input_dims[1]]
            return pos_embed

        if self.cpe_mode:
            if self.training:
                min_scale = math.sqrt(0.1)
                scale = torch.rand(batch_size, 1, 1, device=pos_embed.device) * (1 - min_scale) + min_scale
                aspect_min = math.log(3 / 4)
                aspect_max = -aspect_min
                aspect = torch.exp(torch.rand(batch_size, 1, 1, device=pos_embed.device) * (aspect_max - aspect_min) + aspect_min)

                scale_x = scale * aspect
                scale_y = scale * (1 / aspect)
                scale_xy = torch.stack([scale_x, scale_y], dim=-1).clamp_(0, 1)

                pos_xy = torch.rand(batch_size, 1, 1, 2, device=pos_embed.device) * (1 - scale_xy)

                lin_x = torch.linspace(0, 1, steps=input_dims[1], device=pos_embed.device)[None, None].expand(batch_size, input_dims[0], -1)
                lin_y = torch.linspace(0, 1, steps=input_dims[0], device=pos_embed.device)[None, :, None].expand(batch_size, -1, input_dims[1])

                lin_xy = torch.stack([lin_x, lin_y], dim=-1)

                grid_xy = lin_xy * scale_xy + pos_xy

                # Convert to [-1, 1] range
                grid_xy.mul_(2).sub_(1)

                pos_embed = F.grid_sample(
                    pos_embed.float().expand(batch_size, -1, -1, -1),
                    grid=grid_xy,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True,
                ).to(pos_embed.dtype)
            else:
                # i_rows, i_cols = input_dims
                # p_rows, p_cols = pos_embed.shape[2:]
                # if i_rows <= p_rows and i_cols <= p_cols:
                #     left = (p_cols - i_cols) // 2
                #     top = (p_rows - i_rows) // 2
                #     pos_embed = pos_embed[..., top:top+i_rows, left:left+i_cols]
                # else:
                max_dim = max(input_dims)
                pos_embed = F.interpolate(pos_embed.float(), size=(max_dim, max_dim), align_corners=True, mode='bilinear').to(pos_embed.dtype)

                pos_embed = window_select(pos_embed)
        else:
            pos_embed = window_select(pos_embed)

        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(pos_embed.float(), size=input_dims, align_corners=True, mode='bilinear').to(pos_embed.dtype)

        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        return pos_embed


class Im2Patches(nn.Module):
    """Image patches class
    """

    def __init__(self, patch_size: int):
        """Image patches class

        Args:
            patch_size (int): patch size
        """
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function to transform input image to patches
        """
        if self.patch_size == 1:
            patches = x.flatten(2)
            patches = patches.permute(0, 2, 1)
            return patches

        py = x.shape[-2] // self.patch_size
        px = x.shape[-1] // self.patch_size
        patches = rearrange(x, 'b c (py yy) (px xx) -> b (py px) (c yy xx)',
                            py=py, yy=self.patch_size,
                            px=px, xx=self.patch_size
                            )
        return patches


class ViTPatchLinear(nn.Linear):
    """Patch embedding class
    """

    def __init__(self, patch_size: int, embed_dim: int, **factory):
        """Patch embedding class

        Args:
            patch_size (int): patch size
            embed_dim (int): patch embedding dimension
        """
        super().__init__(
            3 * (patch_size ** 2),
            embed_dim,
            bias=False,
            **factory
        )
        self.patch_size = patch_size

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if self.bias is not None:
            self.bias.data.copy_(state_dict[f'{prefix}bias'])

        chk_weight = state_dict[f'{prefix}weight']
        if chk_weight.shape != self.weight.shape:
            src_patch_size = int(math.sqrt(chk_weight.shape[1] // 3))

            assert (src_patch_size ** 2) * 3 == chk_weight.shape[1], 'Unable to interpolate non-square patch size'

            chk_weight = rearrange(chk_weight, 'b (c h w) -> b c h w', c=3, h=src_patch_size, w=src_patch_size)
            chk_weight = F.interpolate(chk_weight, size=(self.patch_size, self.patch_size), mode='bicubic', align_corners=True, antialias=False)
            chk_weight = rearrange(chk_weight, 'b c h w -> b (c h w)')
        self.weight.data.copy_(chk_weight)
