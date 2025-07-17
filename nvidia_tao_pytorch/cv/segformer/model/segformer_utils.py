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


"""Segformer Model utils"""

import warnings

import torch.nn.functional as F


def resize(input,  # pylint: disable=W0622
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    """Resize the input tensor using the specified method."""
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:  # pylint: disable=R0916
                if ((output_h > 1 and output_w > 1 and input_h > 1 and  # pylint: disable=R0916
                     input_w > 1) and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1)):  # pylint: disable=R0916
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def count_params(net):
    """Utility function to count model parameters."""
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print("Total Parameters: ", total_params)
    print("Trainable Parameters: ", trainable_params)
