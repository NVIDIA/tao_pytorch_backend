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

"""Train Re-Identification model."""
import os
import re

from nvidia_tao_pytorch.core.connectors.checkpoint_connector import TLTCheckpointConnector
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.utilities import update_results_dir

from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.re_identification.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.re_identification.model.pl_reid_model import ReIdentificationModel
from nvidia_tao_pytorch.cv.re_identification.utils.common_utils import check_and_create

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint


def run_experiment(experiment_config,
                   results_dir,
                   key):
    """
    Start the training process.

    This function initializes the re-identification model with the provided experiment configuration.
    It sets up the necessary components such as the status logger and checkpoint callbacks.
    The training is performed using the PyTorch Lightning Trainer.

    Args:
        experiment_config (ExperimentConfig): The experiment configuration containing the model, training, and other parameters.
        results_dir (str): The directory to save the trained model checkpoints and logs.
        key (str): The encryption key for intermediate checkpoints.

    Raises:
        AssertionError: If checkpoint_interval is greater than num_epochs.
    """
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    reid_model = ReIdentificationModel(experiment_config, prepare_for_training=True)

    check_and_create(results_dir)

    num_epochs = experiment_config['train']['num_epochs']
    checkpoint_interval = experiment_config['train']['checkpoint_interval']
    assert checkpoint_interval <= num_epochs, (
        f"Checkpoint interval {checkpoint_interval} > Number of epochs {num_epochs}. "
        f"Please set experiment_config.train.checkpoint_interval < {num_epochs}"
    )
    gpus_ids = experiment_config['train']["gpu_ids"]
    num_gpus = experiment_config['train']['num_gpus']
    grad_clip = experiment_config['train']['grad_clip']

    status_logger_callback = TAOStatusLogger(results_dir, append=True, num_epochs=num_epochs)

    status_logging.set_status_logger(status_logger_callback.logger)

    acc_flag = None
    if num_gpus > 1 or len(gpus_ids) > 1:
        acc_flag = DDPStrategy(find_unused_parameters=False)

    if "swin" in experiment_config["model"]["backbone"]:
        if num_gpus > 1:
            trainer = Trainer(devices=num_gpus,
                              max_epochs=num_epochs,
                              check_val_every_n_epoch=checkpoint_interval,
                              default_root_dir=results_dir,
                              num_sanity_val_steps=0,
                              precision=16,
                              accelerator='gpu',
                              strategy=acc_flag,
                              replace_sampler_ddp=False,
                              sync_batchnorm=True,
                              gradient_clip_val=grad_clip)
        else:
            trainer = Trainer(devices=gpus_ids,
                              max_epochs=num_epochs,
                              check_val_every_n_epoch=checkpoint_interval,
                              default_root_dir=results_dir,
                              num_sanity_val_steps=0,
                              precision=16,
                              accelerator='gpu',
                              strategy=acc_flag,
                              replace_sampler_ddp=False,
                              sync_batchnorm=True,
                              gradient_clip_val=grad_clip)
    elif "resnet" in experiment_config["model"]["backbone"]:
        if num_gpus > 1:
            trainer = Trainer(devices=num_gpus,
                              max_epochs=num_epochs,
                              check_val_every_n_epoch=checkpoint_interval,
                              default_root_dir=results_dir,
                              num_sanity_val_steps=0,
                              accelerator='gpu',
                              strategy=acc_flag,
                              replace_sampler_ddp=False,
                              sync_batchnorm=True,
                              gradient_clip_val=grad_clip)
        else:
            trainer = Trainer(gpus=gpus_ids,
                              max_epochs=num_epochs,
                              check_val_every_n_epoch=checkpoint_interval,
                              default_root_dir=results_dir,
                              num_sanity_val_steps=0,
                              accelerator='gpu',
                              strategy=acc_flag,
                              replace_sampler_ddp=False,
                              sync_batchnorm=True,
                              gradient_clip_val=grad_clip)

    # Overload connector to enable intermediate ckpt encryption and decryption.
    resume_ckpt = experiment_config['train']['resume_training_checkpoint_path']
    trainer._checkpoint_connector = TLTCheckpointConnector(trainer)
    if resume_ckpt is not None:
        trainer._checkpoint_connector.resume_checkpoint_path = resume_ckpt

    # setup checkpointer:
    ModelCheckpoint.FILE_EXTENSION = ".tlt"
    checkpoint_callback = ModelCheckpoint(every_n_epochs=checkpoint_interval,
                                          dirpath=results_dir,
                                          monitor=None,
                                          save_top_k=-1,
                                          filename='reid_model_{epoch:03d}')
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
    trainer.fit(reid_model)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """
    Run the training process.

    This function serves as the entry point for the training script.
    It loads the experiment specification, obfuscates logs, updates the results directory, and calls the 'run_experiment' function.

    Args:
        cfg (ExperimentConfig): The experiment configuration retrieved from the Hydra configuration files.

    Raises:
        KeyboardInterrupt: If the training is interrupted manually.
        SystemExit: If the system or program finishes abruptly.
        Exception: For any other types of exceptions thrown during training.
    """
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
