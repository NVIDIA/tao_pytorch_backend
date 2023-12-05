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

"""Train action recognition model."""
import os
import re

from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.connectors.checkpoint_connector import TLTCheckpointConnector
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.action_recognition.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.action_recognition.model.pl_ar_model import ActionRecognitionModel
from nvidia_tao_pytorch.cv.action_recognition.utils.common_utils import check_and_create

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def run_experiment(experiment_config,
                   results_dir,
                   key):
    """Start the training."""
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    ar_model = ActionRecognitionModel(experiment_config)

    total_epochs = experiment_config['train']['num_epochs']

    check_and_create(results_dir)

    status_logger_callback = TAOStatusLogger(results_dir, append=True, num_epochs=total_epochs)
    status_logging.set_status_logger(status_logger_callback.logger)

    clip_grad = experiment_config['train']['clip_grad_norm']
    gpus_ids = experiment_config['train']["gpu_ids"]
    num_gpus = experiment_config['train']['num_gpus']
    acc_flag = None
    if num_gpus > 1 or len(gpus_ids) > 1:
        acc_flag = "ddp"

    if num_gpus > 1:
        trainer = Trainer(devices=num_gpus,
                          max_epochs=total_epochs,
                          check_val_every_n_epoch=experiment_config['train']['checkpoint_interval'],
                          default_root_dir=results_dir,
                          accelerator='gpu',
                          strategy=acc_flag,
                          gradient_clip_val=clip_grad)
    else:
        trainer = Trainer(gpus=gpus_ids,
                          max_epochs=total_epochs,
                          check_val_every_n_epoch=experiment_config['train']['checkpoint_interval'],
                          default_root_dir=results_dir,
                          accelerator='gpu',
                          strategy=acc_flag,
                          gradient_clip_val=clip_grad)

    # Overload connector to enable intermediate ckpt encryption & decryption.
    resume_ckpt = experiment_config['train']['resume_training_checkpoint_path']
    trainer._checkpoint_connector = TLTCheckpointConnector(trainer)
    if resume_ckpt is not None:
        trainer._checkpoint_connector = TLTCheckpointConnector(trainer, resume_from_checkpoint=resume_ckpt)
    else:
        trainer._checkpoint_connector = TLTCheckpointConnector(trainer)

    ckpt_inter = experiment_config['train']['checkpoint_interval']

    # setup checkpointer:
    ModelCheckpoint.FILE_EXTENSION = ".tlt"
    checkpoint_callback = ModelCheckpoint(every_n_epochs=ckpt_inter,
                                          dirpath=results_dir,
                                          save_on_train_epoch_end=True,
                                          monitor=None,
                                          save_top_k=-1,
                                          filename='ar_model_{epoch:03d}')

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

    trainer.fit(ar_model)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
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
