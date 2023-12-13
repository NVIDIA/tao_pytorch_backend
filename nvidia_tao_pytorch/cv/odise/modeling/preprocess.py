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

import collections.abc
import torch


def batched_input_to_device(batched_inputs, device, exclude=()):

    if isinstance(exclude, str):
        exclude = [exclude]

    if isinstance(batched_inputs, torch.Tensor):
        batch = batched_inputs.to(device, non_blocking=True)
        return batch
    elif isinstance(batched_inputs, collections.abc.Mapping):
        batch = {}
        for k in batched_inputs:
            if k not in exclude:
                batched_inputs[k] = batched_input_to_device(batched_inputs[k], device)
        return batched_inputs

    elif isinstance(batched_inputs, collections.abc.Sequence) and not isinstance(
        batched_inputs, str
    ):
        return [batched_input_to_device(d, device) for d in batched_inputs]
    elif isinstance(batched_inputs, str):
        return batched_inputs
    else:
        raise TypeError(f"Unsupported type {type(batched_inputs)}")
