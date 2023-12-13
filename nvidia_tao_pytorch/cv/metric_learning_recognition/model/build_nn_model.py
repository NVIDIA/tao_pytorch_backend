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

"""The top model builder interface."""
import os

import torch
import torchvision.models as torch_model
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


class MLPSeq(torch.nn.Module):
    """This block implements a series of MLP layers given the input sizes."""

    def __init__(self, layer_sizes, final_relu=False):
        """Initiates the sequential module of MLP layers.

        Args:
            layer_sizes (List[List]): a nested list of MLP layer sizes
            final_relu (Boolean): if True, a ReLu activation layer is added after each MLP layer.
        """
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(torch.nn.ReLU(inplace=True))
            layer_list.append(torch.nn.Linear(input_size, curr_size))
        self.net = torch.nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        """Sequential MLP forward."""
        return self.net(x)


class Baseline(torch.nn.Module):
    """Base model for Metric Learning Recognition model. The model consists of a
    backbone (trunk) and a feature extractor (embedder). The backbone has a softmax
    layer and it would be replaced by an identity layer.
    """

    def __init__(self, trunk, embedder):
        """Initiates the joint modules of the backbone and feature extractors.

        Args:
            embedder (torch.Module): The MLP layers with embedding vector outputs
            trunk (torch.Module): the backbone with fc layer removed
        """
        super().__init__()
        self.embedder = embedder
        self.trunk = trunk

    def forward(self, x):
        """Joint forward function for the backbone and the feature extractor."""
        features_extracted = self.trunk(x)
        output_embeds = self.embedder(features_extracted)
        return output_embeds

    def load_param(self, model_path):
        """Load paramaters for the model from a .pth format pretrained weights.

        Args:
            model_path (str): Model path.
        """
        param_dict = torch.load(model_path)
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]

        for i in param_dict:
            if 'fc' in i:
                continue
            if ("net" in i) or ("bias" in i):
                j = "embedder." + i
            else:
                j = "trunk." + i
            if j in self.state_dict(destination=None).keys():
                self.state_dict(destination=None)[j].copy_(param_dict[i])


def build_model(cfg, checkpoint_to_load=False):
    """Builds metric learning recognition model according to config. If
    `checkpoint_to_load` is True, nothing would be returned as the model is already
    loaded somewhere else. If `checkpoint_to_load` is False, the function would
    do following things: first the model trunk and embedder would be initialized randomly.
    if `model.pretrain_choice` is `imagenet`, the pretrained weights from
    `Torch IMAGENET1K_V2` would be loaded to the trunk. If `model.pretrained_model_path`
    is specified, the pretrained weights from the weights file would be loaded to
    the trunk. If `model.pretrain_choice` is empty and `model.pretrained_model_path`
    is not specified, the trunk would keep its random weights. Notice that the
    embedder would not be loaded with pretrained weights in any case.

    In the end, the embedder and trunk would be combined to a Baseline model and
    the model would be returned.

    Args:
        cfg (DictConfig): Hydra config object.
        checkpoint_to_load (Bool): If True, a checkpoint would be loaded after building the model so the pretrained weights should not be loaded.

    Returns:
        model (torch.Module): the Baseline torch module.
    """
    status_logging.get_status_logger().write(
        message="Constructing model graph...",
        status_level=status_logging.Status.RUNNING)
    model_configs = cfg["model"]
    trunk_model = model_configs["backbone"]
    embed_dim = model_configs["feat_dim"]
    # torchvision model
    load_weights = None
    if not checkpoint_to_load and model_configs["pretrain_choice"] == "imagenet":
        status_logging.get_status_logger().write(
            message="Loading ImageNet pretrained weights to trunk...",
            status_level=status_logging.Status.RUNNING)
        if trunk_model == "resnet_50":
            load_weights = torch_model.ResNet50_Weights.IMAGENET1K_V2
        elif trunk_model == "resnet_101":
            load_weights = torch_model.ResNet101_Weights.IMAGENET1K_V2
        else:
            error_mesage = "`model.backbone` only supports resnet_50 and resnet_101 at this moment."
            status_logging.get_status_logger().write(
                message=error_mesage,
                status_level=status_logging.Status.FAILURE)
            raise ValueError(error_mesage)
    trunk = torch_model.__dict__[trunk_model.replace("_", "")](
        weights=load_weights,
        progress=False)
    trunk_output_size = trunk.fc.in_features
    trunk.fc = torch.nn.Identity()
    embedder = MLPSeq([trunk_output_size, embed_dim])
    model = Baseline(trunk=trunk, embedder=embedder)

    if checkpoint_to_load:
        status_logging.get_status_logger().write(
            message="Skipped loading pretrained model as checkpoint is to load.",
            status_level=status_logging.Status.SKIPPED)
    else:
        status_logging.get_status_logger().write(
            message=f"Loading pretrained model to trunk: {model_configs['pretrained_model_path']}. Embedder pretrain weights loading is not supported now.",
            status_level=status_logging.Status.RUNNING)
        resume_ckpt = model_configs["pretrained_model_path"]
        if resume_ckpt:
            if not os.path.exists(resume_ckpt):
                error_mesage = "`model.pretrained_model_path` file does not exist."
                status_logging.get_status_logger().write(
                    message=error_mesage,
                    status_level=status_logging.Status.FAILURE)
                raise ValueError(error_mesage)
            model.load_param(resume_ckpt)

    return model
