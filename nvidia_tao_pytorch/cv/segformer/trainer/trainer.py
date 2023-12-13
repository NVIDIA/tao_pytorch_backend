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

"""Train Segformer model."""

import random
import warnings
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.iter_based_runner import IterBasedRunner
from mmcv.runner.checkpoint import save_checkpoint
from nvidia_tao_pytorch.cv.segformer.core import DistEvalHook, EvalHook
from nvidia_tao_pytorch.cv.segformer.dataloader.data_utils import build_dataloader, build_dataset
from nvidia_tao_pytorch.cv.segformer.utils import get_root_logger
import os.path as osp
from typing import Optional, Dict


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@RUNNERS.register_module()
class TAOIterBasedRunner(IterBasedRunner):
    """TAO Epoch based runner.

    Overrides  mmcv.runner.epoch_based_runner.EpochBaseRunner to save checkpoints
    without symlinks which requires root access.
    """

    def save_checkpoint(  # type: ignore
            self,
            out_dir: str,
            filename_tmpl: str = 'iter_{}.pth',
            meta: Optional[Dict] = None,
            save_optimizer: bool = True,
            create_symlink: bool = True) -> None:
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
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

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)


def train_segmentor(model,
                    dataset,
                    distributed=True,
                    validate=False,
                    timestamp=None,
                    meta=None,
                    result_dir=None,
                    dm=None,
                    sf_model=None):
    """Launch segmentor training.

    Args:
        model (nn.Module): Model instance
        distributed (Bool): Flag to enable distributed training
        validate (Bool): Flag to enable validation during training.
        dm (Class instance): Dataloader parameters class object.
        sf_model (Model instance): Segformer parameters class object.
        meta (Dict): Meta data like environment variables.

    """
    logger = get_root_logger("INFO")

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            dm.samples_per_gpu,
            dm.workers_per_gpu,
            dm.num_gpus,
            dist=True,
            seed=dm.seed,
            drop_last=True) for ds in dataset
    ]
    # put model on gpus
    if distributed:
        find_unused_parameters = sf_model.find_unused_parameters
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(0), device_ids=[0])

    # build runner
    cfg_optimizer = sf_model.sf_optim_cfg
    lr_config = sf_model.lr_config

    param_wise = cfg_optimizer["paramwise_cfg"]
    cfg_optimizer["paramwise_cfg"] = dict(custom_keys=param_wise)

    optimizer = build_optimizer(model, cfg_optimizer)
    tao_runner = {'type': 'TAOIterBasedRunner', 'max_iters': sf_model.max_iters, "work_dir": result_dir}
    warnings.warn(
        'config is now expected to have a `runner` section, '
        'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        tao_runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=result_dir,
            logger=logger,
            meta=meta))

    log_config = dict(interval=dm.log_interval, hooks=[dict(type='TaoTextLoggerHook', by_epoch=False)])
    checkpoint_config = dict(by_epoch=False, interval=sf_model.checkpoint_interval)
    # register hooks
    checkpoint_config["meta"] = dict(CLASSES=dm.CLASSES, PALETTE=dm.PALETTE)
    runner.register_training_hooks(lr_config=lr_config, optimizer_config={},
                                   checkpoint_config=checkpoint_config, log_config=log_config,
                                   momentum_config=None)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    # # register eval hooks
    if validate:
        val_dataset = build_dataset(dm.val_data, dm.default_args)
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=distributed,
            shuffle=False)
        eval_cfg = dict(interval=sf_model.validation_interval, metric='mIoU')
        eval_cfg['by_epoch'] = False
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority='LOW')
    resume_from = sf_model.resume_ckpt
    load_from = None
    workflow = [('train', 1)]
    if resume_from:
        runner.resume(resume_from)
    elif load_from:
        runner.load_checkpoint(load_from)
    runner.run(data_loaders, workflow)
