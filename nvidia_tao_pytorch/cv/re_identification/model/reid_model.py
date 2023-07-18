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

"""Model builder interface."""
import torch
from nvidia_tao_pytorch.cv.pose_classification.utils.common_utils import patch_decrypt_checkpoint
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook


def load_pretrained_weights(pretrained_backbone_path):
    """Load pretrained weights from the provided path.

    This function decrypts the encrypted state dictionary if necessary, and restructures the keys to remove the
    'model.' prefix from each key (if it exists). If the "state_dict" key is not present in the loaded data,
    it simply returns the loaded data.

    Args:
        pretrained_backbone_path (str): The file path of the pretrained backbone model weights.

    Returns:
        dict: A dictionary containing the model's state dict, with keys adjusted as necessary.

    Raises:
        PermissionError: If the loaded state dict is encrypted but no encryption key is provided.
    """
    temp = torch.load(pretrained_backbone_path,
                      map_location="cpu")

    if temp.get("state_dict_encrypted", False):
        # Retrieve encryption key from TLTPyTorchCookbook.
        key = TLTPyTorchCookbook.get_passphrase()
        if key is None:
            raise PermissionError("Cannot access model state dict without the encryption key")
        temp = patch_decrypt_checkpoint(temp, key)

    if "state_dict" not in temp:
        return temp

    new_state_dict = {}
    for key, value in list(temp["state_dict"].items()):
        if "model" in key:
            new_key = ".".join(key.split(".")[1:])
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict
