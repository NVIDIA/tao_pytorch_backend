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

"""Train CenterPose model."""

import os
import re

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.utilities import update_results_dir
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging

from nvidia_tao_pytorch.cv.centerpose.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.centerpose.model.pl_centerpose_model import CenterPosePlModel
from nvidia_tao_pytorch.cv.centerpose.dataloader.build_data_loader import CPDataModule
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import check_and_create


def run_experiment(experiment_config,
                   results_dir,
                   key):
    """Start the training."""
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)
    # Build dataloader
    dm = CPDataModule(experiment_config.dataset)

    # Load pretrained model as starting point if pretrained path is provided,
    pretrained_path = experiment_config.train.pretrained_model_path
    if pretrained_path is not None:
        pt_model = CenterPosePlModel.load_from_checkpoint(pretrained_path,
                                                          map_location="cpu",
                                                          experiment_spec=experiment_config)
    else:
        pt_model = CenterPosePlModel(experiment_config)

    total_epochs = experiment_config.train.num_epochs

    check_and_create(results_dir)

    status_logger_callback = TAOStatusLogger(
        results_dir,
        append=True,
        num_epochs=total_epochs
    )

    status_logging.set_status_logger(status_logger_callback.logger)

    num_gpus = experiment_config.train.num_gpus
    validation_interval = experiment_config.train.validation_interval
    ckpt_inter = experiment_config.train.checkpoint_interval

    assert ckpt_inter <= total_epochs, (
        f"Checkpoint interval {ckpt_inter} > Number of epochs {total_epochs}."
        f"Please set experiment_config.train.checkpoint_interval < {total_epochs}"
    )

    assert validation_interval <= total_epochs, (
        f"Validation interval {validation_interval} > Number of epochs {total_epochs}."
        f"Please set experiment_config.train.validation_interval < {total_epochs}"
    )

    clip_grad_val = experiment_config.train.clip_grad_val
    is_dry_run = experiment_config.train.is_dry_run

    if experiment_config.train.precision.lower() in ["fp16", "fp32"]:
        precision = int(experiment_config.train.precision.replace("fp", ""))
    else:
        raise NotImplementedError(f"{experiment_config.train.precision} is not supported. Only fp32 and fp16 are supported")

    sync_batchnorm = False
    strategy = None
    if num_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)

    seed_everything(experiment_config.train.randomseed, workers=True)
    trainer = Trainer(devices=num_gpus,
                      max_epochs=total_epochs,
                      check_val_every_n_epoch=validation_interval,
                      default_root_dir=results_dir,
                      accelerator='gpu',
                      strategy=strategy,
                      precision=precision,
                      gradient_clip_val=clip_grad_val,
                      replace_sampler_ddp=False,
                      sync_batchnorm=sync_batchnorm,
                      deterministic=False,
                      fast_dev_run=is_dry_run)

    # Overload connector to enable intermediate ckpt encryption & decryption.
    resume_ckpt = experiment_config.train.resume_training_checkpoint_path

    # setup checkpointer:
    ModelCheckpoint.FILE_EXTENSION = ".pth"
    checkpoint_callback = ModelCheckpoint(every_n_epochs=ckpt_inter,
                                          dirpath=results_dir,
                                          save_on_train_epoch_end=True,
                                          monitor=None,
                                          save_top_k=-1,
                                          filename='centerpose_model_{epoch:03d}')

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
    trainer.fit(pt_model, dm, ckpt_path=resume_ckpt or None)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        cfg = update_results_dir(cfg, task="train")
        run_experiment(experiment_config=cfg,
                       key=cfg.encryption_key,
                       results_dir=cfg.results_dir)
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
