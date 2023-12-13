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

""" Main model file for SegFormer. """

import torch
from omegaconf import OmegaConf


class SFModel(object):  # pylint: disable=too-many-ancestors
    """Pytorch Module for SegFormer Model."""

    def __init__(self, experiment_spec, phase=None, num_classes=None):
        """Init Segformer Model module.
        Args:
            experiment_spec (Dict): Dictionary of the spec parameters.
            phase (str): Indicates train, val or test phase.
            num_classes (int): Number of classes.
        """
        super().__init__()

        self.experiment_spec = experiment_spec
        self.dataset_config = experiment_spec["dataset"]
        self.sf_config = experiment_spec["model"]
        self.phase = phase
        if phase == "train":
            self.sf_optim = experiment_spec["train"]["trainer"]["sf_optim"]
            self.sf_optim_cfg = OmegaConf.to_container(self.sf_optim)
            self.lr_config = OmegaConf.to_container(experiment_spec["train"]["trainer"]["lr_config"])
            self.validation_interval = experiment_spec["train"]["validation_interval"]
            self.find_unused_parameters = experiment_spec["train"]["trainer"]["find_unused_parameters"]

        self.train_cfg = None
        self.test_cfg = OmegaConf.to_container(self.sf_config["test_cfg"])
        self.model_cfg = OmegaConf.to_container(self.sf_config)
        self.model_cfg["pretrained"] = self.model_cfg["pretrained_model_path"]
        self.model_cfg.pop("pretrained_model_path")
        self.model_cfg["type"] = "EncoderDecoder"
        self.model_cfg["decode_head"]["type"] = "SegFormerHead"
        self.model_cfg["backbone"]["style"] = "pytorch"
        self.model_cfg.pop("input_height")
        self.model_cfg.pop("input_width")
        self.backbone = self.model_cfg["backbone"]["type"]

        self.channels_map = {"mit_b0": [32, 64, 160, 256],
                             "fan_tiny_8_p4_hybrid": [128, 256, 192, 192],
                             "fan_large_16_p4_hybrid": [128, 256, 480, 480],
                             "fan_small_12_p4_hybrid": [128, 256, 384, 384],
                             "fan_base_16_p4_hybrid": [128, 256, 448, 448], }
        if self.backbone in self.channels_map:
            self.model_cfg["decode_head"]["in_channels"] = self.channels_map[self.backbone]
        if "fan" in self.backbone:
            self.model_cfg["decode_head"]["channels"] = 256
        self.export = False
        self.model_cfg["backbone"]["export"] = self.export
        self.test_cfg["export"] = self.export
        self.model_cfg["train_cfg"] = self.train_cfg
        if self.test_cfg["mode"] == "whole":
            self.test_cfg.pop("crop_size")
            self.test_cfg.pop("stride")
        self.model_cfg["test_cfg"] = self.test_cfg
        self.model_cfg["decode_head"]["export"] = self.export
        self.model_cfg["decode_head"]["num_classes"] = num_classes
        self.num_classes = num_classes
        self.distributed = experiment_spec["train"]["exp_config"]["distributed"]
        self.checkpoint_loaded = None
        self.tmp_ckpt = None
        self.max_iters = 1
        self.resume_ckpt = None
        self.checkpoint_interval = 1000

    def _convert_batchnorm(self, module):
        """ Convert Sync BN during Export."""
        module_output = module
        if isinstance(module, torch.nn.SyncBatchNorm):
            module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                                 module.momentum, module.affine,
                                                 module.track_running_stats)
            if module.affine:
                module_output.weight.data = module.weight.data.clone().detach()
                module_output.bias.data = module.bias.data.clone().detach()
                # keep requires_grad unchanged
                module_output.weight.requires_grad = module.weight.requires_grad
                module_output.bias.requires_grad = module.bias.requires_grad
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(name, self._convert_batchnorm(child))
        del module
        return module_output
