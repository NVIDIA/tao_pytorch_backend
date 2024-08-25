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

"""Common utilties for PTL."""

import glob
import os
import shutil
import struct
import torch

from eff.core.codec import encrypt_stream
from nvidia_tao_pytorch.core.connectors.checkpoint_connector import decrypt_checkpoint

# Define 1MB for filesize calculation.
MB = 1 << 20


def get_num_trainable_elements(model):
    """Get number of trainable model elements.

    Args:
        model (ptl.module): Pytorch lightning module.

    Return:
        size (int): Number of elements in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_file_size(model_path):
    """Get the size of the model.

    Args:
        model_path (str): UNIX path to the model.

    Returns:
        file_size (float): File size in MB.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file wasn't found at {model_path}")
    file_size = os.path.getsize(model_path) / MB
    return file_size


def update_results_dir(cfg, task):
    """Update global results_dir based on task.results_dir.

    This function should be called at the beginning of a pipeline script.

    Args:
        cfg (Hydra config): Config object loaded by Hydra
        task (str): TAO pipeline name

    Returns:
        Updated cfg
    """
    if cfg[task]['results_dir']:
        cfg['results_dir'] = cfg[task]['results_dir']
    elif cfg['results_dir']:
        cfg['results_dir'] = os.path.join(cfg['results_dir'], task)
        cfg[task]['results_dir'] = cfg['results_dir']
    else:
        raise ValueError("You need to set at least one of following fields: results_dir, {mode}.results_dir")
    print(f"{task.capitalize()} results will be saved at: {cfg['results_dir']}")

    return cfg


# TODO: do we still need this?
def get_last_generated_file(folder_path, extension="txt"):
    """Returns the last generated file in the folder.

    Args:
        folder_path (str): path to the folder
        extension (str): file extension
    """
    files = glob.glob(os.path.join(folder_path, f"*.{extension}"))
    return max(files, key=os.path.getmtime, default=None)


def get_latest_checkpoint(folder_path):
    """Returns the latest checkpoint in the (possibly remote) folder.

    Args:
        folder_path (str): path to the folder
    """
    # The ModelCheckpoint callback creates a file "{model_name}_latest.pth"
    ckpt = glob.glob(os.path.join(folder_path, "*_latest.pth"))
    if ckpt:
        return os.path.realpath(ckpt[0])
    return None


def patch_decrypt_checkpoint(checkpoint, key):
    """Decrypt checkpoint to work when using a multi-GPU trained model in a single-GPU environment.

    Args:
        checkpoint (dict): The encrypted checkpoint.
        key (str): The decryption key.

    Returns:
        dict: The patched decrypted checkpoint.

    """
    from functools import partial
    legacy_load = torch.load
    torch.load = partial(legacy_load, map_location="cpu")

    checkpoint = decrypt_checkpoint(checkpoint, key)

    torch.load = legacy_load

    # set the encrypted status to be False when it is decrypted
    checkpoint["state_dict_encrypted"] = False

    return checkpoint


def check_and_create(d):
    """
    Create a directory if it does not already exist.

    Args:
        d (str): The path of the directory to create.
    """
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def check_and_delete(d):
    """Delete a directory."""
    if os.path.isdir(d):
        shutil.rmtree(d)


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


def encrypt_onnx(tmp_file_name, output_file_name, key):
    """Encrypt the onnx model"""
    with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                           "wb") as open_encoded_file:
        # set the input name magic number
        open_encoded_file.write(struct.pack("<i", 0))

        encrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )
