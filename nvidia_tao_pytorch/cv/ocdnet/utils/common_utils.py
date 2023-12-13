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

"""Utils for ocdnet."""

import os
import torch
import struct
import json
import numpy as np
import tempfile
from eff.codec import encrypt_stream


def encrypt_onnx(tmp_file_name, output_file_name, key):
    """Encrypt the onnx model."""
    with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                           "wb") as open_encoded_file:
        # set the input name magic number
        open_encoded_file.write(struct.pack("<i", 0))

        encrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def encrypt_pytorch(tmp_file_name, output_file_name, key):
    """Encrypt the pytorch model"""
    with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                           "wb") as open_encoded_file:
        encrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def save_checkpoint(state, filename, key):
    """Save the checkpoint."""
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)
    handle, temp_name = tempfile.mkstemp(".tlt")
    os.close(handle)
    torch.save(state, temp_name)
    encrypt_pytorch(temp_name, filename, key)
    os.remove(temp_name)


def check_and_create(d):
    """Create a directory."""
    if not os.path.isdir(d):
        os.makedirs(d)


def load_json_from_file(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_np_to_file(file_path, data):
    """Write Numpy array to file."""
    np.save(file=file_path, arr=data, allow_pickle=False)


def data_to_device(data):
    """Transfer data to GPU."""
    if isinstance(data, list):
        cuda_data = []
        for item in data:
            cuda_item = item.cuda(non_blocking=True)
            cuda_data.append(cuda_item)
    else:
        cuda_data = data.cuda(non_blocking=True)

    return cuda_data
