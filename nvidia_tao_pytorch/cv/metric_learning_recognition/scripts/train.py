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
import re

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.core.utilities import get_last_generated_file
from nvidia_tao_pytorch.cv.metric_learning_recognition.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.metric_learning_recognition.model.pl_ml_recog_model import MLRecogModel
from nvidia_tao_pytorch.cv.metric_learning_recognition.utils.decorators import monitor_status


def run_experiment(experiment_config):
    """Starts the training.

    Args:
        experiment_config (DictConfig): Configuration dictionary
        results_dir (str): Output directory

    """
    # no need to check `else` as it's verified in the decorator already
    if experiment_config['train']["results_dir"]:
        results_dir = experiment_config['train']["results_dir"]
    elif experiment_config["results_dir"]:
        results_dir = os.path.join(experiment_config["results_dir"], "train")

    status_logger_callback = TAOStatusLogger(
        results_dir, append=True,
        num_epochs=experiment_config['train']['num_epochs'])
    status_logging.set_status_logger(status_logger_callback.logger)

    metric_learning_recognition = MLRecogModel(
        experiment_config,
        results_dir,
        subtask="train")
    total_epochs = experiment_config['train']['num_epochs']

    clip_grad = experiment_config['train']['clip_grad_norm']
    gpus_ids = experiment_config['train']["gpu_ids"]
    acc_flag = None
    if len(gpus_ids) > 1:
        acc_flag = "ddp"

    ckpt_inter = experiment_config['train']['checkpoint_interval']
    trainer = Trainer(gpus=gpus_ids,
                      max_epochs=total_epochs,
                      check_val_every_n_epoch=ckpt_inter,
                      val_check_interval=0.99,
                      default_root_dir=results_dir,
                      num_sanity_val_steps=0,
                      accelerator='gpu',
                      strategy=acc_flag,
                      gradient_clip_val=clip_grad)

    # setup checkpointer:
    ModelCheckpoint.FILE_EXTENSION = ".pth"
    checkpoint_callback = ModelCheckpoint(every_n_epochs=ckpt_inter,
                                          dirpath=results_dir,
                                          monitor=None,
                                          save_top_k=-1,
                                          filename='ml_model_{epoch:03d}',
                                          save_on_train_epoch_end=True)

    trainer.callbacks.append(checkpoint_callback)

    if experiment_config['train']['resume_training_checkpoint_path']:
        resume_training_checkpoint_path = experiment_config['train']['resume_training_checkpoint_path']

    resume_training_checkpoint_path = get_last_generated_file(results_dir, extension="pth")  # None if no pth files found

    if resume_training_checkpoint_path:
        status_logging.get_status_logger().write(
            message=f"Resuming training from checkpoint: {resume_training_checkpoint_path}",
            status_level=status_logging.Status.STARTED
        )
        resumed_epoch = re.search('epoch=(\\d+)', resume_training_checkpoint_path)
        if resumed_epoch:
            resumed_epoch = int(resumed_epoch.group(1))
        else:
            resumed_epoch = 0
        status_logger_callback.epoch_counter = resumed_epoch + 1  # make sure callback epoch matches resumed epoch

    trainer.callbacks.append(status_logger_callback)

    trainer.fit(metric_learning_recognition, ckpt_path=resume_training_checkpoint_path)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
@monitor_status(mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process.

    Args:
        cfg (DictConfig): Hydra config object.
    """
    obfuscate_logs(cfg)

    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
