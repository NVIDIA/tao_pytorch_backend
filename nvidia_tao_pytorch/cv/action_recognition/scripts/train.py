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

from nvidia_tao_pytorch.core.connectors.checkpoint_connector import TLTCheckpointConnector
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.action_recognition.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.action_recognition.dataloader.pl_ar_data_module import ARDataModule
from nvidia_tao_pytorch.cv.action_recognition.model.pl_ar_model import ActionRecognitionModel
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment

from pytorch_lightning import Trainer


def run_experiment(experiment_config, key):
    """Start the training."""
    results_dir, resume_ckpt, gpus, ptl_loggers = initialize_train_experiment(experiment_config, key)

    dm = ARDataModule(experiment_config)

    ar_model = ActionRecognitionModel(experiment_config, dm)

    total_epochs = experiment_config['train']['num_epochs']
    validation_interval = experiment_config['train']['validation_interval']
    clip_grad = experiment_config['train']['clip_grad_norm']

    trainer = Trainer(logger=ptl_loggers,
                      devices=gpus,
                      max_epochs=total_epochs,
                      check_val_every_n_epoch=validation_interval,
                      default_root_dir=results_dir,
                      accelerator='gpu',
                      strategy='auto',
                      enable_checkpointing=False,
                      gradient_clip_val=clip_grad)

    # Overload connector to enable intermediate ckpt encryption & decryption.
    trainer._checkpoint_connector = TLTCheckpointConnector(trainer)

    trainer.fit(ar_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name='Action Recognition', mode='train')
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
