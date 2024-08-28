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

"""Train metric-learning recognition model."""

import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.cv.metric_learning_recognition.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.metric_learning_recognition.dataloader.pl_ml_data_module import MLDataModule
from nvidia_tao_pytorch.cv.metric_learning_recognition.model.pl_ml_recog_model import MLRecogModel


def run_experiment(experiment_config):
    """Starts the training.

    Args:
        experiment_config (DictConfig): Configuration dictionary
        results_dir (str): Output directory

    """
    results_dir, resume_ckpt, gpus, ptl_loggers = initialize_train_experiment(experiment_config)

    # update experiment_config in Trainer
    experiment_config['train']['resume_training_checkpoint_path'] = resume_ckpt

    dm = MLDataModule(experiment_config)

    metric_learning_recognition = MLRecogModel(
        experiment_config,
        results_dir,
        dm,
        subtask="train")

    total_epochs = experiment_config['train']['num_epochs']
    clip_grad = experiment_config['train']['clip_grad_norm']
    val_inter = experiment_config['train']['validation_interval']

    trainer = Trainer(logger=ptl_loggers,
                      devices=gpus,
                      max_epochs=total_epochs,
                      check_val_every_n_epoch=val_inter,
                      val_check_interval=0.99,
                      default_root_dir=results_dir,
                      num_sanity_val_steps=0,
                      accelerator='gpu',
                      strategy='auto',
                      enable_checkpointing=False,
                      gradient_clip_val=clip_grad)

    trainer.fit(metric_learning_recognition, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
@monitor_status(name="Metric Learning Recognition", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process.

    Args:
        cfg (DictConfig): Hydra config object.
    """
    obfuscate_logs(cfg)

    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
