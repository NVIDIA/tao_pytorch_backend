# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MAL/blob/main/LICENSE
"""MAL evaluation script."""

import os
import warnings

import torch
from pytorch_lightning import Trainer

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.mal.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.mal.datasets.pl_data_module import WSISDataModule
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.mal.models.mal import MAL
from nvidia_tao_pytorch.cv.mal.utils.config_utils import update_config
warnings.filterwarnings("ignore")
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="eval", schema=ExperimentConfig
)
def run_evaluation(cfg: ExperimentConfig) -> None:
    """Run evaluation."""
    cfg = update_config(cfg, 'evaluate')
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Set status logging
    status_file = os.path.join(cfg.results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            append=True
        )
    )
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting MAL evaluation"
    )

    # gpu indices
    if len(cfg.gpu_ids) == 0:
        cfg.gpu_ids = list(range(torch.cuda.device_count()))

    cfg.train.lr = 0
    cfg.train.min_lr = 0
    cfg.train.batch_size = cfg.evaluate.batch_size
    num_workers = len(cfg.gpu_ids) * cfg.dataset.num_workers_per_gpu

    data_loader = WSISDataModule(
        num_workers=num_workers,
        load_train=False,
        load_val=True, cfg=cfg)

    model = MAL(
        cfg=cfg, num_iter_per_epoch=1,
        categories=data_loader._val_data_loader.dataset.coco.dataset['categories'])

    trainer = Trainer(
        devices=1,
        gpus=cfg.gpu_ids,
        num_nodes=cfg.num_nodes,
        strategy=cfg.strategy,
        accelerator='gpu', max_epochs=-1,
        precision=16, check_val_every_n_epoch=1,
        resume_from_checkpoint=cfg.checkpoint,
        accumulate_grad_batches=1)

    trainer.validate(model, ckpt_path=cfg.checkpoint, dataloaders=data_loader.val_dataloader())


if __name__ == '__main__':
    try:
        run_evaluation()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Evaluation finished successfully"
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Evaluation was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
