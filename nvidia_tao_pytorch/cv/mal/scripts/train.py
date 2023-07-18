# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MAL/blob/main/LICENSE
"""MAL training script."""
import logging
import os
import glob
import warnings

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.mal.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.mal.datasets.pl_data_module import WSISDataModule
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.mal.models.mal import MAL
from nvidia_tao_pytorch.cv.mal.utils.config_utils import update_config
warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="train", schema=ExperimentConfig
)
def run_experiment(cfg: ExperimentConfig) -> None:
    """Run training experiment."""
    # set random seed
    seed_everything(cfg.train.seed)
    cfg = update_config(cfg, 'train')
    os.makedirs(cfg.results_dir, exist_ok=True)

    status_logger_callback = TAOStatusLogger(
        cfg.results_dir,
        append=True,
        num_epochs=cfg.train.num_epochs
    )

    status_logging.set_status_logger(status_logger_callback.logger)

    # gpu indices
    if len(cfg.gpu_ids) == 0:
        cfg.gpu_ids = list(range(torch.cuda.device_count()))

    cfg.train.lr = cfg.train.lr * len(cfg.gpu_ids) * cfg.train.batch_size
    cfg.train.min_lr = cfg.train.lr * cfg.train.min_lr_rate
    num_workers = len(cfg.gpu_ids) * cfg.dataset.num_workers_per_gpu
    logger.info("Setting up dataloader...")
    data_loader = WSISDataModule(
        num_workers=num_workers,
        load_train=True,
        load_val=True, cfg=cfg)

    num_iter_per_epoch = len(data_loader.train_dataloader())

    ModelCheckpoint.FILE_EXTENSION = ".pth"
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.results_dir,
        filename=f'{cfg.model.arch.replace("/", "-")}' + '{epoch:03d}',
        save_top_k=-1,
        every_n_epochs=cfg.train.save_every_k_epoch,
        save_weights_only=True,
        save_last=False)

    resume_checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.results_dir,
        filename=f'{cfg.model.arch.replace("/", "-")}_resume',
        save_top_k=1,
        every_n_epochs=cfg.train.save_every_k_epoch,
        save_last=False)

    resume_ckpt = sorted(glob.glob(
        os.path.join(cfg.results_dir, f'{cfg.model.arch.replace("/", "-")}_resume*')))
    if resume_ckpt:
        resume_ckpt = resume_ckpt[-1]
        logger.info(f"Training will resume from {resume_ckpt}.")
        cfg.checkpoint = None
    logger.info("Building MAL models...")
    model = MAL(
        cfg=cfg, num_iter_per_epoch=num_iter_per_epoch,
        categories=data_loader._train_data_loader.dataset.coco.dataset['categories'])

    trainer = Trainer(
        gpus=cfg.gpu_ids,
        num_nodes=cfg.num_nodes,
        strategy=cfg.strategy,
        devices=None,
        callbacks=[status_logger_callback, checkpoint_callback, resume_checkpoint_callback],
        accelerator='gpu',
        default_root_dir=cfg.results_dir,
        max_epochs=cfg.train.num_epochs,
        precision=16 if cfg.train.use_amp else 32,
        check_val_every_n_epoch=cfg.train.val_interval,
        accumulate_grad_batches=cfg.train.accum_grad_batches)

    trainer.fit(model, data_loader, ckpt_path=resume_ckpt or None)


if __name__ == '__main__':
    try:
        run_experiment()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Training finished successfully"
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
