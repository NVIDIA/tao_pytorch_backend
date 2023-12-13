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

import os
import os.path as osp
import tempfile
import time
import torch
import torch.distributed as dist
from abc import abstractmethod
import dataclasses
from omegaconf import OmegaConf
from nvidia_tao_pytorch.core.mmlab.mmclassification.model_params_mapping import map_params
import mmcv
from mmcv.runner import get_dist_info, load_checkpoint
from mmcls.apis.test import collect_results_gpu, collect_results_cpu
from mmcls.models import build_classifier


ROOT_DIR = os.getenv("NV_TLT_PYTORCH_TOP", os.getcwd())


class MMClsConfig(object):
    """Classification Config Class to convert Hydra config to MMcls config"""

    def __init__(self,
                 config,
                 phase="train"):
        """Init Function."""
        self.config = dataclasses.asdict(OmegaConf.to_object(config))
        self.phase = phase
        self.update_config(phase=phase)

    def update_config(self, phase="train"):
        """ Function to update hydra config to mmlab based config"""
        self.config = self.update_dataset_config(self.config)
        self.config = self.update_model_config(self.config)
        if phase == "train":
            self.config = self.update_train_params_config(self.config)

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

    @abstractmethod
    def update_dataset_config(self, cfg):
        """Update the dataset config"""
        #  Update Dataset config
        #  Update train data pipeline
        img_norm_cfg = cfg["dataset"]["img_norm_cfg"]
        pipeline = cfg["dataset"]["data"]["train"]["pipeline"]  # Augmentations
        pipeline_updated = [dict(type='LoadImageFromFile')] + pipeline + [dict(type='Normalize', **img_norm_cfg),
                                                                          dict(type='ImageToTensor', keys=['img']),
                                                                          dict(type='ToTensor', keys=['gt_label']),
                                                                          dict(type='Collect', keys=['img', 'gt_label'])]
        cfg["dataset"]["data"]["train"]["pipeline"] = pipeline_updated

        #  Update test pipeline
        test_pipeline = []
        test_pipeline_tmp = cfg["dataset"]["data"]["test"]["pipeline"]
        # Convert resize size to tuple fro mmcv loader
        for aug in test_pipeline_tmp:
            if aug["type"] == "Resize":
                aug["size"] = tuple(aug["size"])
            test_pipeline.append(aug)
        test_pipeline_updated = [dict(type='LoadImageFromFile')] + test_pipeline + [dict(type='Normalize', **img_norm_cfg),
                                                                                    dict(type='ImageToTensor', keys=['img']),
                                                                                    dict(type='Collect', keys=['img'])]
        cfg["dataset"]["data"]["test"]["pipeline"] = test_pipeline_updated
        cfg["dataset"]["data"]["val"]["pipeline"] = test_pipeline_updated

        return cfg

    @abstractmethod
    def update_model_config(self, cfg):
        """Update the model config"""
        #  Update Model Config
        #  Head Update
        #  Tok should be tuple. Hydra converts it to list by default
        cfg["model"]["head"]["topk"] = tuple(cfg["model"]["head"]["topk"])

        #  init_cfg should be removed if checkpoint is none
        if not cfg["model"]["init_cfg"]["checkpoint"]:
            cfg["model"].pop("init_cfg")

        #  Update head params from the map json
        map_params_head = map_params.get("head", None)
        cfg["model"]["head"] = self.assign_arch_specific_params(cfg["model"]["head"], map_params_head, cfg["model"]["backbone"]["type"])
        map_params_head = map_params.get("backbone", None)

        #  Update backbone params from the map json
        cfg["model"]["backbone"] = self.assign_arch_specific_params(cfg["model"]["backbone"], map_params_head, cfg["model"]["backbone"]["type"])
        if cfg["model"]["neck"]:  # Neck config is not must. Hence we do this check
            map_params_neck = map_params.get("neck", None)
            cfg["model"]["neck"] = self.assign_arch_specific_params(cfg["model"]["neck"], map_params_neck, cfg["model"]["backbone"]["type"])
        cfg["model"]["head"] = self.update_custom_args(cfg["model"]["head"])
        cfg["model"]["backbone"] = self.update_custom_args(cfg["model"]["backbone"])

        return cfg

    def update_train_params_config(self, cfg):
        """Update train parameters"""
        #  Update Train Params
        paramwise_cfg = cfg["train"]["train_config"].get("paramwise_cfg", None)
        if paramwise_cfg:
            cfg["train"]["train_config"]["optim_cfg"].update(paramwise_cfg)
        return cfg


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
            raise OSError((f'The tmpdir {tmpdir} already exists.',
                           ' Since tmpdir will be deleted after testing,',
                           ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    dist.barrier()
    img_names = []
    for _, data in enumerate(data_loader):
        img_names += [f["filename"] for f in data["img_metas"].data[0]]
        with torch.no_grad():
            result = model(return_loss=False, **data)
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results, img_names


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
