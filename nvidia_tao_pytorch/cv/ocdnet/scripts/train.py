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
#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
"""Train OCDnet model."""
import os
import re

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.ocdnet.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocdnet.model.pl_ocd_model import OCDnetModel
from omegaconf import OmegaConf


def run_experiment(tmp_experiment_config,
                   results_dir):
    """Start the training."""
    if tmp_experiment_config.train.results_dir is not None:
        results_dir = tmp_experiment_config.train.results_dir
    else:
        results_dir = os.path.join(tmp_experiment_config.results_dir, "train")
        tmp_experiment_config.train.results_dir = results_dir

    os.makedirs(results_dir, exist_ok=True)

    experiment_config = OmegaConf.to_container(tmp_experiment_config)
    ocd_model = OCDnetModel(experiment_config)

    total_epochs = experiment_config['train']['num_epochs']
    assert total_epochs != experiment_config['train']['lr_scheduler']['args']['warmup_epoch'], "num_epochs should not be same as warmup_epoch."
    val_inter = experiment_config['train']['validation_interval']
    clip_grad = experiment_config['train']['trainer']['clip_grad_norm']
    num_gpus = experiment_config["num_gpus"]
    activation_checkpoint = experiment_config['model']['activation_checkpoint']
    distributed_strategy = experiment_config['train']['distributed_strategy']
    is_dry_run = experiment_config['train']['is_dry_run']

    if experiment_config['train']['precision'].lower() in ["fp16", "fp32"]:
        precision = int(experiment_config['train']['precision'].lower().replace("fp", ""))
    else:
        raise NotImplementedError(f"{experiment_config['train']['precision'].lower()} is not supported. Only fp32 and fp16 are supported")

    status_logger_callback = TAOStatusLogger(
        results_dir,
        append=True,
        num_epochs=total_epochs
    )
    status_logging.set_status_logger(status_logger_callback.logger)

    sync_batchnorm = False
    strategy = None
    if num_gpus > 1:
        # By default find_unused_parameters is set to True in Lightning for backward compatibility
        # This introduces extra overhead and can't work with activation checkpointing
        # Ref: https://pytorch-lightning.readthedocs.io/en/1.8.5/advanced/model_parallel.html#when-using-ddp-strategies-set-find-unused-parameters-false
        # TODO: Starting from PTL 2.0, find_usued_parameters is set to False by default
        if distributed_strategy.lower() == "ddp" and activation_checkpoint:
            strategy = DDPStrategy(find_unused_parameters=False)
        elif distributed_strategy.lower() == "ddp" and not activation_checkpoint:
            strategy = 'ddp'
        elif distributed_strategy.lower() == "ddp_sharded":
            strategy = 'ddp_sharded'
            # Override to FP16 for ddp_sharded as there's an error with FP32 during Positional Embedding forward pass
            print("Overriding Precision to FP16 for ddp_sharded")
            precision = 16
        elif distributed_strategy.lower() == "deepspeed_stage_3_offload":
            strategy = 'deepspeed_stage_3_offload'
            print("Overriding Precision to FP16 for deepspeed_stage_3_offload")
            precision = 16
        else:
            raise NotImplementedError(f"{distributed_strategy} is not implemented. Only ddp , ddp_sharded and deepspeed are supported")

        if "fan" in experiment_config['model']['backbone']:
            print("Setting sync batch norm")
            sync_batchnorm = True

    trainer = Trainer(devices=num_gpus,
                      max_epochs=total_epochs,
                      check_val_every_n_epoch=val_inter,
                      default_root_dir=results_dir,
                      enable_checkpointing=False,
                      accelerator="gpu",
                      strategy=strategy,
                      gradient_clip_val=clip_grad,
                      num_sanity_val_steps=0,
                      callbacks=None,
                      precision=precision,
                      sync_batchnorm=sync_batchnorm,
                      fast_dev_run=is_dry_run)

    ckpt_inter = experiment_config['train']['checkpoint_interval']
    ModelCheckpoint.FILE_EXTENSION = ".pth"
    checkpoint_callback = ModelCheckpoint(every_n_epochs=ckpt_inter,
                                          dirpath=results_dir,
                                          save_on_train_epoch_end=True,
                                          monitor=None,
                                          save_top_k=-1,
                                          filename='ocd_model_{epoch:03d}')
    resume_ckpt = experiment_config['train']['resume_training_checkpoint_path']
    if resume_ckpt:
        status_logging.get_status_logger().write(
            message=f"Resuming training from checkpoint: {resume_ckpt}",
            status_level=status_logging.Status.STARTED
        )
        resumed_epoch = re.search('epoch=(\\d+)', resume_ckpt)
        if resumed_epoch:
            resumed_epoch = int(resumed_epoch.group(1))
        else:
            resumed_epoch = 0
        status_logger_callback.epoch_counter = resumed_epoch + 1  # make sure callback epoch matches resumed epoch

    trainer.callbacks.append(status_logger_callback)
    trainer.callbacks.append(checkpoint_callback)

    if resume_ckpt and resume_ckpt.endswith(".pth"):
        print(f'Resume training model from {resume_ckpt}')
        trainer.fit(ocd_model, ckpt_path=resume_ckpt)
    else:
        trainer.fit(ocd_model)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        run_experiment(tmp_experiment_config=cfg,
                       results_dir=cfg.train.results_dir)
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


if __name__ == "__main__":
    main()
