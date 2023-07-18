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

""" MSDeformAttn modules. """

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
import os

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from nvidia_tao_pytorch.cv.deformable_detr.model.ops.functions import MSDeformAttnFunction, load_ops


def _is_power_of_2(n):
    """Check if n is power of 2.

    Args:
        n (int): input

    Returns:
        Boolean on if n is power of 2 or not.
    """
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Multi-Scale Deformable Attention Constructor.

        Args:
            d_model (int): hidden dimension
            n_levels (int): number of feature levels
            n_heads (int): number of attention heads
            n_points (int): number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()
        # load custom ops
        ops_dir = os.path.dirname(os.path.abspath(__file__))
        lib_name = "MultiScaleDeformableAttention.cpython-38-x86_64-linux-gnu.so"
        load_ops(ops_dir, lib_name)

    def _reset_parameters(self):
        """Reset parameters."""
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, export=False):
        """Forward function.

        Args:
            query (torch.Tensor): (N, Length_{query}, C)
            reference_points (torch.Tensor): (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                             or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
            input_flatten (torch.Tensor): (N, sum_{l=0}^{L-1} H_l cdot W_l, C)
            input_spatial_shapes (torch.Tensor): (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            input_level_start_index (torch.Tensor): (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
            input_padding_mask (torch.Tensor): (N, sum_{l=0}^{L-1} H_l cdot W_l), True for padding elements, False for non-padding elements

        Returns:
            output (torch.Tensor): (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1],
                                            input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        input_spatial_shapes = input_spatial_shapes.long()
        input_level_start_index = input_level_start_index.long()

        if export:
            if torch.cuda.is_available() and value.is_cuda:
                output = torch.ops.nvidia.MultiscaleDeformableAttnPlugin_TRT(
                    value, input_spatial_shapes, input_level_start_index,
                    sampling_locations, attention_weights)
            else:
                # CPU implementation of multi-scale deformable attention
                # Note that this implementation uses GridSample operator which requires
                # opset version >= 16 and is much slower in TensorRT
                warnings.warn("PyTorch native implementation of multi-scale deformable attention is being used. "
                              "Expect slower inference performance until TensorRT further optimizes GridSample.")
                output = multi_scale_deformable_attn_pytorch(
                    value, input_spatial_shapes, sampling_locations, attention_weights
                )
        else:
            if torch.cuda.is_available() and value.is_cuda:
                # For mixed precision training
                if value.dtype == torch.float16:
                    output = MSDeformAttnFunction.apply(
                        value.to(torch.float32), input_spatial_shapes,
                        input_level_start_index, sampling_locations.to(torch.float32),
                        attention_weights, self.im2col_step)
                    output = output.to(torch.float16)
                else:
                    output = MSDeformAttnFunction.apply(
                        value, input_spatial_shapes, input_level_start_index,
                        sampling_locations, attention_weights, self.im2col_step)
            else:
                # CPU implementation of multi-scale deformable attention
                output = multi_scale_deformable_attn_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)

        output = output.view(N, Len_q, self.d_model)
        output = self.output_proj(output)
        return output
