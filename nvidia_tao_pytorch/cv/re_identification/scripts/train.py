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

from nvidia_tao_pytorch.core.connectors.checkpoint_connector import TLTCheckpointConnector
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.cv.re_identification.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.re_identification.dataloader.pl_reid_data_module import REIDDataModule
from nvidia_tao_pytorch.cv.re_identification.model.pl_reid_model import ReIdentificationModel

from pytorch_lightning import Trainer


def run_experiment(experiment_config, key):
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
    results_dir, resume_ckpt, gpus, ptl_loggers = initialize_train_experiment(experiment_config, key)

    dm = REIDDataModule(experiment_config)
    reid_model = ReIdentificationModel(experiment_config, prepare_for_training=True)

    num_epochs = experiment_config['train']['num_epochs']
    validation_interval = experiment_config['train']['validation_interval']
    grad_clip = experiment_config['train']['grad_clip']

    acc_flag = 'auto'
    if len(gpus) > 1:
        acc_flag = 'ddp_find_unused_parameters_true'

    trainer = Trainer(logger=ptl_loggers,
                      devices=len(gpus),
                      max_epochs=num_epochs,
                      check_val_every_n_epoch=validation_interval,
                      default_root_dir=results_dir,
                      num_sanity_val_steps=0,
                      val_check_interval=0.99,
                      precision='16-mixed',
                      accelerator='gpu',
                      strategy=acc_flag,
                      use_distributed_sampler=False,
                      sync_batchnorm=True,
                      enable_checkpointing=False,
                      gradient_clip_val=grad_clip)

    # Overload connector to enable intermediate ckpt encryption and decryption.
    trainer._checkpoint_connector = TLTCheckpointConnector(trainer)

    trainer.fit(reid_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="ReIdentification", mode="train")
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
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
