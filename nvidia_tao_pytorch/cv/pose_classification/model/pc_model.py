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

"""Model builder interface and joint model."""
import torch
from nvidia_tao_pytorch.cv.pose_classification.model.st_gcn import st_gcn
from nvidia_tao_pytorch.cv.pose_classification.utils.common_utils import patch_decrypt_checkpoint
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook


def load_pretrained_weights(pretrained_model_path):
    """
    Load the pre-trained weights from a checkpoint.

    This function loads the weights from a specified checkpoint. If the weights are encrypted,
    the function retrieves the encryption key from the TLTPyTorchCookbook and decrypts the weights.

    Args:
        pretrained_model_path (str): The path to the pre-trained model checkpoint.

    Returns:
        dict: A dictionary containing the state of the model.

    Raises:
        PermissionError: If the weights are encrypted and the encryption key is not available.
    """
    temp = torch.load(pretrained_model_path,
                      map_location="cpu")

    if temp.get("state_dict_encrypted", False):
        # Retrieve encryption key from TLTPyTorchCookbook.
        key = TLTPyTorchCookbook.get_passphrase()
        if key is None:
            raise PermissionError("Cannot access model state dict without the encryption key")
        temp = patch_decrypt_checkpoint(temp, key)

    state_dict = {}
    for key, value in list(temp["state_dict"].items()):
        if "model" in key:
            new_key = ".".join(key.split(".")[1:])
            state_dict[new_key] = value
        else:
            state_dict[key] = value

    return state_dict


def get_basemodel(pretrained_model_path,
                  input_channels,
                  num_classes,
                  graph_layout,
                  graph_strategy,
                  edge_importance_weighting=True,
                  data_bn=True,
                  **kwargs):
    """
    Get the base model for ST-GCN.

    This function creates an instance of the ST-GCN model, and if a pre-trained model is provided,
    it loads the pre-trained weights into the model.

    Args:
        pretrained_model_path (str): The path to the pre-trained model checkpoint.
        input_channels (int): The number of input channels.
        num_classes (int): The number of classes in the dataset.
        graph_layout (str): The graph layout to be used in the model.
        graph_strategy (str): The graph strategy to be used in the model.
        edge_importance_weighting (bool, optional): Whether to use edge importance weighting. Defaults to True.
        data_bn (bool, optional): Whether to use batch normalization. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.nn.Module: The created ST-GCN model.
    """
    if pretrained_model_path:
        print("loading trained weights from {}".format(
            pretrained_model_path))
        pretrained_weights = load_pretrained_weights(pretrained_model_path)
    else:
        pretrained_weights = None

    model = st_gcn(pretrained_weights=pretrained_weights,
                   input_channels=input_channels,
                   num_classes=num_classes,
                   graph_layout=graph_layout,
                   graph_strategy=graph_strategy,
                   edge_importance_weighting=edge_importance_weighting,
                   data_bn=data_bn,
                   **kwargs)

    return model
