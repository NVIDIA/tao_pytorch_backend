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

"""Train Optical Inspection Siamese Network model."""
import os
import re
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utilities import get_last_generated_file
from nvidia_tao_pytorch.cv.optical_inspection.config.default_config import OIExperimentConfig
from nvidia_tao_pytorch.cv.optical_inspection.model.pl_oi_model import OpticalInspectionModel
from nvidia_tao_pytorch.cv.optical_inspection.utils.common_utils import check_and_create

CHECKPOINT_FILE_EXT = "pth"


def run_experiment(experiment_config,
                   results_dir,
                   key):
    """Start the training."""
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    # Load pretrained model as starting point if pretrained path is provided,
    pretrained_path = experiment_config.train.pretrained_model_path
    if pretrained_path is not None:
        oi_model = OpticalInspectionModel.load_from_checkpoint(pretrained_path,
                                                               map_location="cpu",
                                                               experiment_spec=experiment_config)
    else:
        oi_model = OpticalInspectionModel(experiment_config)

    check_and_create(results_dir)

    total_epochs = experiment_config['train']['num_epochs']
    clip_grad = experiment_config['train']['clip_grad_norm']
    gpu_idx = experiment_config['train']['gpu_ids']
    num_gpus = experiment_config['train']['num_gpus']
    # Handling multiGPU with num_gpus.
    if num_gpus != len(gpu_idx):
        logging.warning(
            "Number of gpus [{num_gpus}] != [{gpu_ids}].".format(
                num_gpus=num_gpus,
                gpu_ids=gpu_idx
            )
        )
        num_gpus = max(num_gpus, len(gpu_idx))
        gpu_idx = range(num_gpus) if len(gpu_idx) != num_gpus else gpu_idx
        logging.info(
            "Setting the num_gpus: {num_gpus} and train.gpu_ids: {gpu_ids}".format(
                gpu_ids=experiment_config['train']['gpu_ids'],
                num_gpus=gpu_idx
            )
        )
    validation_interval = experiment_config.train.validation_interval
    checkpoint_interval = experiment_config.train.checkpoint_interval
    enable_tensorboard = experiment_config.train.tensorboard.enabled
    assert checkpoint_interval <= total_epochs, (
        f"Checkpoint interval {checkpoint_interval} > Number of epochs {total_epochs}."
        f"Please set experiment_config.train.checkpoint_interval < {total_epochs}"
    )
    assert validation_interval <= total_epochs, (
        f"Validation interval {validation_interval} > Number of epochs {total_epochs}."
        f"Please set experiment_config.train.validation_interval < {total_epochs}"
    )

    status_logger_callback = TAOStatusLogger(results_dir, append=True, num_epochs=total_epochs)
    status_logging.set_status_logger(status_logger_callback.logger)

    trainer_kwargs = {}
    if enable_tensorboard:
        trainer_kwargs["logger"] = TensorBoardLogger(
            save_dir=results_dir
        )
        infrequent_logging_frequency = experiment_config.train.tensorboard.infrequent_logging_frequency
        assert max(0, infrequent_logging_frequency) <= total_epochs, (
            f"infrequent_logging_frequency {infrequent_logging_frequency} must be < num_epochs {total_epochs}"
        )
        logging.info("Tensorboard logging enabled.")
    else:
        logging.info("Tensorboard logging disabled.")
    acc_flag = None
    if num_gpus > 1:
        acc_flag = "ddp"
    trainer = Trainer(
        devices=num_gpus,
        gpus=gpu_idx,
        max_epochs=total_epochs,
        check_val_every_n_epoch=validation_interval,
        default_root_dir=results_dir,
        accelerator='gpu',
        strategy=acc_flag,
        gradient_clip_val=clip_grad,
        **trainer_kwargs
    )
    resume_ckpt = None
    if experiment_config['train']['resume_training_checkpoint_path']:
        resume_ckpt = experiment_config['train']['resume_training_checkpoint_path']
    else:
        # Get the latest checkpoint file to resume training from by default.
        resume_ckpt = get_last_generated_file(
            results_dir,
            extension=CHECKPOINT_FILE_EXT
        )
        logging.info("Setting resume checkpoint to {}".format(resume_ckpt))

    logging.info(
        "Results directory {} Checkpoint Interval {}".format(results_dir, checkpoint_interval)
    )
    # Setup model checkpoint callback.
    ModelCheckpoint.FILE_EXTENSION = f".{CHECKPOINT_FILE_EXT}"
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=checkpoint_interval,
        dirpath=results_dir,
        save_on_train_epoch_end=True,
        monitor=None,
        save_top_k=-1,
        filename='oi_model_{epoch:03d}'
    )
    trainer.callbacks.append(checkpoint_callback)

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
        status_logger_callback.epoch_counter = resumed_epoch + 1
    trainer.callbacks.append(status_logger_callback)
    trainer.fit(oi_model, ckpt_path=resume_ckpt or None)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment", schema=OIExperimentConfig
)
def main(cfg: OIExperimentConfig) -> None:
    """Run the training process."""
    try:
        if cfg.train.results_dir is not None:
            results_dir = cfg.train.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "train")
        run_experiment(experiment_config=cfg,
                       results_dir=results_dir,
                       key=cfg.encryption_key)
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


if __name__ == "__main__":
    main()
