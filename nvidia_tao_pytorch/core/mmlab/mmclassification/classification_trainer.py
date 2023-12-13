# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Original source taken from https://github.com/open-mmlab/mmclassification

# Copyright 2019 OpenMMLAB

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer Class for Classification."""

from mmcv.runner import build_optimizer, build_runner, DistSamplerSeedHook
from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcls.core import DistEvalHook, DistOptimizerHook
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import get_root_logger
from nvidia_tao_pytorch.core.mmlab.common.base_trainer import MMCVTrainer
from nvidia_tao_pytorch.core.mmlab.mmclassification.utils import load_model
import os.path as osp


@RUNNERS.register_module()
class TAOEpochBasedRunner(EpochBasedRunner):
    """TAO Epoch based runner.

    Overrides  mmcv.runner.epoch_based_runner.EpochBaseRunner to save checkpoints
    without symlinks which requires root access.
    """

    def __init__(self, *args, **kwargs):
        """Init Function."""
        super(TAOEpochBasedRunner, self).__init__(*args, **kwargs)

    def save_checkpoint(self, out_dir, filename_tmpl='epoch_{}.pth', save_optimizer=True,
                        meta=None, create_symlink=False):
        """Checkpoint saver
        Args:
            out_dir (str): Output dir to save checkpoints
            filename_tmpl (str): Checkpoint saving template
            save_optimizer (bool): Flag to whether to save optimizer states
            meta (Dict): Dictionary that has the checkpoint meta variables
            create_symlink (bool): Flag whether to create sym link to the latest checkpoint

        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)


class MMClsTrainer(MMCVTrainer):
    """MMClassification Trainer."""

    def __init__(self, *args, **kwargs):
        """Init Function."""
        super(MMClsTrainer, self).__init__(*args, **kwargs)

    def set_up_runner(self):
        """Function to Set Up Runner"""
        # build runner
        optimizer = self.train_cfg["optimizer"]
        lr_config = self.train_cfg["lr_config"]
        optimizer_config = self.train_cfg["optimizer_config"]
        runner_config = self.train_cfg["runner"]
        optimizer = build_optimizer(self.model, optimizer)
        logger = get_root_logger(osp.join(self.result_dir, "INFO"))
        self.runner = build_runner(
            runner_config,
            default_args=dict(model=self.model, batch_processor=None, optimizer=optimizer,
                              work_dir=self.result_dir, logger=logger, meta=self.meta))
        if optimizer_config:
            optimizer_config = DistOptimizerHook(**optimizer_config)

        log_config = dict(interval=self.train_cfg["logging"]["interval"],
                          hooks=[dict(type='MMClsTaoTextLoggerHook')])
        checkpoint_config = dict(interval=self.train_cfg["checkpoint_config"]["interval"])
        custom_hooks = self.train_cfg["custom_hooks"]
        # register hooks
        self.runner.register_training_hooks(lr_config=lr_config, optimizer_config=optimizer_config,
                                            checkpoint_config=checkpoint_config, log_config=log_config,
                                            momentum_config=None,
                                            custom_hooks_config=custom_hooks)

        # Register Dist hook for sampler
        if runner_config['type'] == 'TAOEpochBasedRunner':
            self.runner.register_hook(DistSamplerSeedHook())

        # an ugly walkaround to make the .log and .log.json filenames the same
        self.runner.timestamp = self.timestamp
        # # register eval hooks
        resume_from = self.train_cfg["resume_training_checkpoint_path"]
        load_from = self.train_cfg["load_from"]
        if resume_from:
            modified_ckpt = load_model(resume_from, return_ckpt=True)
            self.runner.resume(modified_ckpt)
        elif load_from:
            self.runner.load_checkpoint(load_from)
        return self.runner

    def set_up_data_loaders(self):
        """Function to generate dataloaders"""
        # prepare data loaders
        dataset = self.dataset if isinstance(self.dataset, (list, tuple)) else [self.dataset]
        self.data_loaders = [
            build_dataloader(
                ds,
                self.dataset_cfg["data"]["samples_per_gpu"],
                self.dataset_cfg["data"]["workers_per_gpu"],
                self.cfg["train"]["num_gpus"],
                dist=True,
                seed=self.cfg["train"]["exp_config"]["manual_seed"],
                drop_last=True) for ds in dataset
        ]
        return self.data_loaders

    def validate_runner(self):
        """Function to Add validation hook to training"""
        val_dataset = build_dataset(self.dataset_cfg["data"]["val"], dict(test_mode=True))
        # The specific dataloader settings
        val_dataloader = dict(samples_per_gpu=self.dataset_cfg["data"]["samples_per_gpu"], workers_per_gpu=self.dataset_cfg["data"]["workers_per_gpu"])
        sampler_cfg = self.dataset_cfg["sampler"]
        loader_cfg = dict(num_gpus=self.cfg["train"]["num_gpus"], dist=True, seed=self.cfg["train"]["exp_config"]["manual_seed"],
                          round_up=True, sampler_cfg=sampler_cfg)
        val_loader_cfg = {**loader_cfg, 'shuffle': False,  # Not shuffle by default
                          'sampler_cfg': None,  # Not use sampler by default
                          'drop_last': False,  # Not drop last by default
                          **val_dataloader}
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
        eval_cfg = dict(interval=self.train_cfg["evaluation"]["interval"], metric="accuracy", metric_options={"topk": (1, 1)})
        eval_cfg['by_epoch'] = self.train_cfg["runner"]['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook
        # `EvalHook` needs to be executed after `IterTimerHook`.
        # Otherwise, it will cause a bug if use `IterBasedRunner`.
        # Refers to https://github.com/open-mmlab/mmcv/issues/1261
        self.runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority='LOW')
