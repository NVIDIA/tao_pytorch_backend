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

"""Utils for action recognition"""

import os
import csv
import torch
import shutil
import struct
from eff.core.codec import encrypt_stream
from nvidia_tao_pytorch.core.connectors.checkpoint_connector import decrypt_checkpoint


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


def check_and_create(d):
    """Create a directory."""
    if not os.path.isdir(d):
        os.makedirs(d)


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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        """Init"""
        self.reset()

    def reset(self):
        """reset parameters."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update accuracy."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, checkpoint, model_best):
    """Naive checkpoint saver."""
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)


def record_train_info(info, filename):
    """Naive log information."""
    str_log = "train_loss: {} val_loss: {} train_acc@1: {} val_acc@1: {} lr: {}".format(
        info['train_loss'],
        info['val_loss'],
        info['train_acc@1'],
        info['val_acc@1'],
        info['lr'])
    print(str_log)
    column_names = ['epoch', 'train_loss', 'val_loss', 'train_acc@1', 'val_acc@1', 'lr']

    if not os.path.isfile(filename):
        with open(filename, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=column_names)
            writer.writeheader()
            writer.writerow(info)
    else:  # else it exists so append without writing the header
        with open(filename, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=column_names)
            writer.writerow(info)
