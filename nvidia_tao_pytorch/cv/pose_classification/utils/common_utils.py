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

"""Utils for pose classification."""

import os
import torch
import struct
import json
import numpy as np
from eff.core.codec import encrypt_stream
from nvidia_tao_pytorch.core.connectors.checkpoint_connector import decrypt_checkpoint


def patch_decrypt_checkpoint(checkpoint, key):
    """
    Decrypt the given checkpoint and adjust it for single-GPU usage.

    This function temporarily modifies the torch.load function to ensure the checkpoint is loaded onto the CPU.
    It decrypts the checkpoint using the provided key, and then resets torch.load back to its original state.
    The 'state_dict_encrypted' field in the checkpoint is also set to False to indicate it has been decrypted.

    Args:
        checkpoint (dict): The checkpoint to decrypt.
        key (str): The decryption key.

    Returns:
        dict: The decrypted checkpoint.
    """
    from functools import partial
    legacy_load = torch.load
    torch.load = partial(legacy_load, map_location="cpu")

    checkpoint = decrypt_checkpoint(checkpoint, key)

    torch.load = legacy_load

    # set the encrypted status to be False when it is decrypted
    checkpoint["state_dict_encrypted"] = False

    return checkpoint


def encrypt_onnx(tmp_file_name, output_file_name, key):
    """
    Encrypt the onnx model.

    The function reads an ONNX model from a file, encrypts it using the provided key,
    and writes the encrypted model to a new file.

    Args:
        tmp_file_name (str): The path to the file containing the ONNX model to encrypt.
        output_file_name (str): The path where the encrypted ONNX model should be written.
        key (str): The encryption key.
    """
    with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                           "wb") as open_encoded_file:
        # set the input name magic number
        open_encoded_file.write(struct.pack("<i", 0))

        encrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def check_and_create(d):
    """
    Create a directory if it does not already exist.

    Args:
        d (str): The path of the directory to create.
    """
    if not os.path.isdir(d):
        os.makedirs(d)


def load_json_from_file(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path of the JSON file to load data from.

    Returns:
        dict: The data loaded from the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_np_to_file(file_path, data):
    """
    Write a Numpy array to a file.

    Args:
        file_path (str): The path where the file should be written.
        data (numpy.ndarray): The Numpy array to write to the file.
    """
    np.save(file=file_path, arr=data, allow_pickle=False)


def data_to_device(data):
    """
    Transfer data to GPU.

    If the data is a list, each item in the list is moved to the GPU individually. Otherwise, the entire data
    object is moved to the GPU.

    Args:
        data (torch.Tensor or list of torch.Tensor): The data to move to the GPU.

    Returns:
        torch.Tensor or list of torch.Tensor: The data on the GPU.
    """
    if isinstance(data, list):
        cuda_data = []
        for item in data:
            cuda_item = item.cuda(non_blocking=True)
            cuda_data.append(cuda_item)
    else:
        cuda_data = data.cuda(non_blocking=True)

    return cuda_data
