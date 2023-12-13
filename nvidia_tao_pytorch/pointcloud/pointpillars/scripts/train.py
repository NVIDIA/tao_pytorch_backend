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

"""Training script for PointPillars."""
import argparse
import datetime
import os
from pathlib import Path
import shutil

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.config import (
    cfg,
    cfg_from_yaml_file,
)
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.datasets import build_dataloader
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.models import (
    build_model_and_optimizer,
    model_fn_decorator
)
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils
from nvidia_tao_pytorch.pointcloud.pointpillars.tools.train_utils.optimization import build_optimizer, build_scheduler
from nvidia_tao_pytorch.pointcloud.pointpillars.tools.train_utils.train_utils import train_model


def parse_config():
    """Argument Parser."""
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument("--output_dir", type=str, required=False, default=None, help="output checkpoint directory.")
    parser.add_argument("--gpus", "-g", type=int, default=1, help="Number of GPUs to run the training")
    parser.add_argument("--key", "-k", type=str, required=True, help="Encryption key")
    args = parser.parse_args()
    cfg_from_yaml_file(expand_path(args.cfg_file), cfg)
    return args, cfg


def main():
    """Main function."""
    args, cfg = parse_config()
    args.workers = cfg.dataset.num_workers
    args.sync_bn = cfg.model.sync_bn
    args.batch_size = cfg.train.batch_size
    args.epochs = cfg.train.num_epochs
    args.ckpt = cfg.train.resume_training_checkpoint_path
    args.pretrained_model = cfg.model.pretrained_model_path
    args.pruned_model = cfg.train.pruned_model_path
    args.tcp_port = cfg.train.tcp_port
    args.fix_random_seed = cfg.train.random_seed
    args.ckpt_save_interval = cfg.train.checkpoint_interval
    args.max_ckpt_save_num = cfg.train.max_checkpoint_save_num
    args.merge_all_iters_to_one_epoch = cfg.train.merge_all_iters_to_one_epoch
    if args.gpus == 1:
        dist_train = False
        total_gpus = 1
    elif args.gpus > 1:
        total_gpus, cfg.LOCAL_RANK = common_utils.init_dist_pytorch(
            args.local_rank, backend='nccl'
        )
        dist_train = True
    else:
        raise ValueError(f"Number of GPUs should be >=1, got: {args.gpus}")
    if args.fix_random_seed is not None:
        common_utils.set_random_seed(args.fix_random_seed)
    if args.output_dir is None:
        if cfg.results_dir is None:
            raise OSError("Either provide results_dir in config file or provide output_dir as a CLI argument")
        else:
            args.output_dir = cfg.results_dir
    output_dir = Path(expand_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    # Set up status logger
    status_file = os.path.join(str(output_dir), "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            append=True
        )
    )
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting PointPillars training"
    )
    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    if cfg.LOCAL_RANK == 0:
        # os.system('cp %s %s' % (args.cfg_file, output_dir))
        os.makedirs(expand_path(output_dir), exist_ok=True)
        shutil.copyfile(args.cfg_file, os.path.join(output_dir, os.path.basename(args.cfg_file)))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.dataset,
        class_names=cfg.class_names,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )
    model, opt_state, start_epoch, it = build_model_and_optimizer(
        cfg.model,
        len(cfg.class_names),
        train_set,
        args.pruned_model,
        args.ckpt,
        args.pretrained_model,
        dist_train,
        logger,
        args.key
    )
    optimizer = build_optimizer(model, cfg.train)
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    # Reload optimizer states after model moved to cuda() so the optimizer
    # states are also moved to the same device
    optimizer.load_state_dict(optimizer.state_dict())
    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    # Build LR scheduler
    last_epoch = -1
    if start_epoch > 0:
        last_epoch = start_epoch + 1
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.train
    )
    logger.info('**********************Start training**********************')
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.train,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        status_logging=status_logging,
        ckpt_save_dir=output_dir,
        key=args.key,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
    )

    logger.info('**********************End training**********************')


if __name__ == '__main__':
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Training finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Training was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
