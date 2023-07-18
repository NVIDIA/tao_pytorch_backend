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
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
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

    status_logger_callback = TAOStatusLogger(
        results_dir,
        append=True,
        num_epochs=total_epochs
    )
    status_logging.set_status_logger(status_logger_callback.logger)

    strategy = None
    if num_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)

    trainer = Trainer(devices=num_gpus,
                      max_epochs=total_epochs,
                      check_val_every_n_epoch=val_inter,
                      default_root_dir=results_dir,
                      enable_checkpointing=False,
                      accelerator="gpu",
                      strategy=strategy,
                      gradient_clip_val=clip_grad,
                      num_sanity_val_steps=0,
                      callbacks=None
                      )

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
