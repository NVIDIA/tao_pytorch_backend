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

"""
Train OCRNet script.
"""
import os
import re
import random
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.ocrnet.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocrnet.model.pl_ocrnet import OCRNetModel


def run_experiment(experiment_spec: ExperimentConfig):
    """run experiment."""
    if experiment_spec.train.results_dir is not None:
        results_dir = experiment_spec.train.results_dir
    else:
        results_dir = os.path.join(experiment_spec.results_dir, "train")
        experiment_spec.train.results_dir = results_dir

    total_epochs = experiment_spec.train.num_epochs
    os.makedirs(f'{results_dir}', exist_ok=True)
    manual_seed = experiment_spec.train.seed
    import torch
    import torch.backends.cudnn as cudnn
    import numpy as np

    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    status_logger_callback = TAOStatusLogger(results_dir,
                                             append=True,
                                             num_epochs=total_epochs)

    status_logging.set_status_logger(status_logger_callback.logger)

    ocrnet_model = OCRNetModel(experiment_spec)
    clip_grad = experiment_spec.train.clip_grad_norm
    gpus_ids = experiment_spec.train.gpu_ids
    distributed_strategy = None
    if len(gpus_ids) > 1:
        distributed_strategy = experiment_spec.train.distributed_strategy

    val_inter = experiment_spec.train.validation_interval
    trainer = Trainer(gpus=gpus_ids,
                      max_epochs=total_epochs,
                      check_val_every_n_epoch=val_inter,
                      default_root_dir=results_dir,
                      enable_checkpointing=False,
                      strategy=distributed_strategy,
                      accelerator='gpu',
                      num_sanity_val_steps=0,
                      gradient_clip_val=clip_grad)

    ckpt_inter = experiment_spec.train.checkpoint_interval
    ModelCheckpoint.FILE_EXTENSION = ".pth"
    checkpoint_callback = ModelCheckpoint(every_n_epochs=ckpt_inter,
                                          dirpath=results_dir,
                                          save_on_train_epoch_end=True,
                                          monitor=None,
                                          save_top_k=-1,
                                          filename='ocrnet_{epoch:03d}')
    resume_ckpt = experiment_spec['train']['resume_training_checkpoint_path']
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

    trainer.fit(ocrnet_model)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        run_experiment(experiment_spec=cfg)
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


if __name__ == '__main__':
    main()
