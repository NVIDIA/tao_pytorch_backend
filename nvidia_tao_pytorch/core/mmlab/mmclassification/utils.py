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

"""Utils Function"""

import copy
import os
import tempfile
from abc import abstractmethod
import dataclasses
from omegaconf import OmegaConf

import torch

from nvidia_tao_pytorch.core.mmlab.mmclassification.model_params_mapping import map_params, map_input_lr_head, map_clip_model_cfg

from mmengine.runner.checkpoint import load_checkpoint
from mmpretrain.models import build_classifier

ROOT_DIR = os.getenv("NV_TLT_PYTORCH_TOP", os.getcwd())


class MMPretrainConfig(object):
    """
    Classification Config Class to convert Hydra config to MMcls config
    """

    PHASE_MAP = {"train": "train",
                 "evaluate": "val",
                 "inference": "test"}

    def __init__(self,
                 config,
                 phase="train"):
        """Init Function."""
        self.config = dataclasses.asdict(OmegaConf.to_object(config))
        self.updated_config = {}
        self.phase = phase
        self.update_config(phase=phase)

    def update_config(self, phase="train"):
        """ Function to update hydra config to mmlab based config"""
        self.update_env()
        self.update_dataset_config()
        self.update_model_config()
        if phase == "train":
            self.update_train_params_config()

        if self.config["model"]["head"]["type"] == "LogisticRegressionHead":
            # Currently, only support Resize for preprocess pipeline for LRHead
            # Other online augmentations are not supported for now due to LRHead apply FeatureExtractor to get feature
            # instead of utilizing dataloader
            self.update_pipeline_config()

        self.updated_config["val_cfg"] = dict()
        self.updated_config["test_cfg"] = dict()

    def update_env(self):
        """Function to update env variables"""
        exp_config = self.config[self.phase]["exp_config"]
        self.updated_config["env_cfg"] = exp_config["env_config"]
        self.updated_config["randomness"] = {"seed": exp_config["manual_seed"], "deterministic": exp_config["deterministic"]}
        self.updated_config["default_scope"] = "mmpretrain"
        vis_backends = [dict(type='LocalVisBackend')]
        self.updated_config["log_level"] = "INFO"
        self.updated_config["vis_backends"] = vis_backends
        self.updated_config["visualizer"] = dict(type='UniversalVisualizer', vis_backends=vis_backends)
        self.updated_config["launcher"] = "pytorch"

    @abstractmethod
    def update_custom_args(self, cfg):
        """Function to upate any custom args"""
        custom_args = cfg.get("custom_args", None)
        if custom_args:
            cfg.update(custom_args)
        cfg.pop("custom_args")
        return cfg

    @abstractmethod
    def assign_arch_specific_params(self, cfg, map_params, backbone_type):
        """Function to assign arch specific parameters from the PARAMS json
        Args:
            map_params (Dict): Dictionary that has the mapping of the various classes.
            backbone_type (str): Backbone type.
        """
        if cfg and map_params:
            params = map_params.keys()
            for param in params:
                orig = cfg[param]
                map_params_tmp = map_params[param]
                cfg[param] = map_params_tmp.get(backbone_type, orig)
        return cfg

    def get_dataloader_config(self, dataset_config, phase):
        """Function to get dataloader config"""
        dataloader_config = {}
        dataloader_config["batch_size"] = dataset_config["data"]["samples_per_gpu"]
        dataloader_config["num_workers"] = dataset_config["data"]["workers_per_gpu"]
        dataloader_config["pin_memory"] = dataset_config["pin_memory"]
        dataloader_config["sampler"] = dataset_config["sampler"]
        dataloader_config["collate_fn"] = dataset_config["collate_fn"]
        dataloader_config["dataset"] = dataset_config["data"][phase]
        dataloader_config["dataset"]["pipeline"] = [{"type": "LoadImageFromFile"}] + dataloader_config["dataset"]["pipeline"] + [{"type": "PackInputs"}]

        return dataloader_config

    @abstractmethod
    def update_dataset_config(self):
        """Update the dataset config"""
        #  Update Dataset config
        dataset_config = self.config["dataset"]
        head_config = self.config["model"]["head"]
        topk = tuple(self.config["model"]["head"].pop("topk"))  # topk is not supported in MMPretrain in head

        self.updated_config["dataset_type"] = dataset_config["data"][self.PHASE_MAP[self.phase]]["type"]
        self.updated_config["data_preprocessor"] = dataset_config["img_norm_cfg"]
        self.updated_config["data_preprocessor"]["num_classes"] = head_config["num_classes"]
        self.updated_config["train_dataloader"] = self.get_dataloader_config(dataset_config, "train")
        self.updated_config["val_dataloader"] = self.get_dataloader_config(dataset_config, "val")
        self.updated_config["test_dataloader"] = self.get_dataloader_config(dataset_config, "test")
        self.updated_config["val_evaluator"] = {"type": "Accuracy", "topk": tuple(topk)}
        self.updated_config["test_evaluator"] = {"type": "Accuracy", "topk": tuple(topk)}

    @abstractmethod
    def update_model_config(self):
        """Update the model config"""
        #  Update Model Config
        #  Head Update
        #  Tok should be tuple. Hydra converts it to list by default
        self.updated_config["model"] = copy.deepcopy(self.config["model"])
        if self.updated_config["model"]["head"]["type"] == "FANLinearClsHead":  # For Backward compatibility
            self.updated_config["model"]["head"]["type"] = "TAOLinearClsHead"

        #  init_cfg should be removed if checkpoint is none
        if self.updated_config["model"]["init_cfg"]["checkpoint"]:
            self.updated_config["model"]["backbone"]["init_cfg"] = self.updated_config["model"]["init_cfg"]
        self.updated_config["model"].pop("init_cfg", None)

        # Update head params from the map json
        map_params_head = map_params.get("head", None)
        self.updated_config["model"]["head"].pop("lr_head")
        if self.updated_config["model"]["head"]["type"] == "LogisticRegressionHead":
            self.updated_config["model"]["head"]["type"] = "TAOLinearClsHead"
            self.updated_config["model"]["head"]["binary"] = self.updated_config["data_preprocessor"]["num_classes"] == 2  # user binary under the hood

        self.updated_config["model"]["head"] = self.update_custom_args(self.updated_config["model"]["head"])
        self.updated_config["model"]["backbone"] = self.update_custom_args(self.updated_config["model"]["backbone"])
        if self.updated_config["model"]["backbone"]["type"] == "open_clip":
            bb_type = self.updated_config["model"]["backbone"]["model_name"]
        else:
            bb_type = self.updated_config["model"]["backbone"]["type"]
        self.updated_config["model"]["head"] = self.assign_arch_specific_params(self.updated_config["model"]["head"], map_params_head, bb_type)

        #  Update backbone params from the map json
        map_params_backbone = map_params.get("backbone", None)
        self.updated_config["model"]["backbone"] = self.assign_arch_specific_params(self.updated_config["model"]["backbone"], map_params_backbone, bb_type)

        #  Update neck params from the map json
        if self.updated_config["model"]["neck"]:  # Neck config is not must. Hence we do this check
            map_params_neck = map_params.get("neck", None)
            self.updated_config["model"]["neck"] = self.assign_arch_specific_params(self.updated_config["model"]["neck"], map_params_neck, bb_type)

        # Update model config for open_clip
        if self.updated_config["model"]["backbone"]["type"] == "open_clip":
            self.updated_config["model"]["backbone"]["model_cfg"] = map_clip_model_cfg.get(bb_type)

    def get_updated_optimizer(self, cfg):
        """Get the updated optimizer"""
        optim_wrapper = {}
        optim_params = cfg["optimizer"]
        optim_wrapper = {"optimizer": optim_params}
        if cfg["optimizer_config"]["grad_clip"]:
            optim_wrapper["clip_grad"] = {"max_norm": float(cfg["optimizer_config"]["grad_clip"]["max_norm"])}
        optim_wrapper["paramwise_cfg"] = cfg["paramwise_cfg"]

        return optim_wrapper

    def update_train_params_config(self):
        """Update train parameters"""
        #  Update Train Params
        train_param_config = self.config["train"]["train_config"]
        self.updated_config["default_hooks"] = train_param_config["default_hooks"]
        self.updated_config["default_hooks"]["checkpoint"]["interval"] = train_param_config["checkpoint_config"]["interval"]
        self.updated_config["default_hooks"]["logger"]["type"] = "TaoTextLoggerHook"
        self.updated_config["default_hooks"]["logger"]["interval"] = train_param_config["logging"]["interval"]
        self.updated_config["auto_scale_lr"] = {"base_batch_size": train_param_config["runner"]["auto_scale_lr_bs"]}
        self.updated_config["train_cfg"] = {"by_epoch": True, "max_epochs": train_param_config["runner"]["max_epochs"], "val_interval": train_param_config["evaluation"]["interval"]}
        self.updated_config["optim_wrapper"] = self.get_updated_optimizer(train_param_config)
        self.updated_config["param_scheduler"] = [train_param_config["lr_config"]]
        self.updated_config["load_from"] = train_param_config["load_from"]
        self.updated_config["resume"] = train_param_config["resume"]
        self.updated_config["custom_hooks"] = train_param_config["custom_hooks"]
        self.updated_config["find_unused_parameters"] = train_param_config["find_unused_parameters"]

    def update_pipeline_config(self):
        """Update pipeline config.
        Currently, this is a special handling for lr head training and inference
        """
        resize_scale = map_input_lr_head.get(self.config["model"]["backbone"]["type"])
        if not resize_scale:
            for pipeline in self.config["dataset"]["data"][self.PHASE_MAP[self.phase]]["pipeline"]:
                if pipeline["type"] == "Resize" or pipeline["type"] == "RandomResizedCrop":
                    resize_scale = pipeline["scale"]
                    break
        pipeline = {"type": "Resize", "scale": resize_scale}
        self.updated_config["test_dataloader"]["dataset"]["pipeline"] = [{"type": "LoadImageFromFile"}] + \
            [pipeline] + [{"type": "PackInputs"}]


def load_model(model_path, mmcls_config=None, return_ckpt=False):
    """Load state dict from the model path

    Args:
        mmcls_config (Dict): Dictionary containing MMCLs config parameters.
        return_ckpt (Bool): Bool whether to return the loaded checkpoint path
    Returns:
        Returns the loaded model instance.
    """
    # Forcing delete to close.
    temp = tempfile.NamedTemporaryFile(
        suffix='.pth',
        delete=True
    )
    tmp_model_path = temp.name

    # Remove EMA related items from the state_dict
    new_state_dict = {}
    checkpoint = torch.load(model_path)
    for k, v in checkpoint["state_dict"].items():
        if 'ema_' not in k:
            new_state_dict[k] = v
    checkpoint['state_dict'] = new_state_dict
    torch.save(checkpoint, tmp_model_path)
    if return_ckpt:
        return tmp_model_path
    model_to_test = build_classifier(mmcls_config["model"])
    _ = load_checkpoint(model_to_test, tmp_model_path, map_location='cpu')
    temp.close()

    return model_to_test
