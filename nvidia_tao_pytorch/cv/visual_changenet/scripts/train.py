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

"""Train Visual ChangeNet model."""

import os

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.core.tlt_logging import logging, obfuscate_logs
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule
from nvidia_tao_pytorch.cv.visual_changenet.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.dataloader.pl_changenet_data_module import CNDataModule
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlSegment
from nvidia_tao_pytorch.cv.visual_changenet.classification.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlClassifier

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


# TODO: @zbhat modify this to get model with best val accuracy for evaluation
# TODO: @seanf this isn't used anywhere, but is it still necessary given how we now save checkpoints?
def get_latest_tlt_model(results_dir):
    """Utility function to return the latest tlt model in a dir."""
    trainable_ckpts = [int(item.split('.')[0].split('_')[1]) for item in os.listdir(results_dir)
                       if item.endswith(".tlt")]
    num_ckpts = len(trainable_ckpts)
    if num_ckpts == 0:
        return None
    latest_step = sorted(trainable_ckpts, reverse=True)[0]
    latest_checkpoint = expand_path(os.path.join(results_dir, f"iter_{latest_step}.tlt"))
    if not os.path.isfile(latest_checkpoint):
        raise FileNotFoundError("Checkpoint file not found at {}")
    return latest_checkpoint


def run_experiment(experiment_config, key):
    """Start the training."""
    results_dir, resume_ckpt, gpus, ptl_loggers = initialize_train_experiment(experiment_config, key)

    task = experiment_config.task
    num_nodes = experiment_config.train.num_nodes
    total_epochs = experiment_config.train.num_epochs
    validation_interval = experiment_config.train.validation_interval
    enable_tensorboard = experiment_config.train.tensorboard.enabled

    # Load pretrained model as starting point if pretrained path is provided
    pretrained_path = experiment_config.train.pretrained_model_path

    precision = '32-true'
    sync_batchnorm = False
    trainer_kwargs = {}

    assert task in ['segment', 'classify'], "Visual ChangeNet only supports 'segment' and 'classify' tasks."
    if task == 'classify':

        dm = OIDataModule(experiment_config, changenet=True)

        if pretrained_path:
            model = ChangeNetPlClassifier.load_from_checkpoint(pretrained_path,
                                                               map_location="cpu",
                                                               experiment_spec=experiment_config,
                                                               dm=dm)
        else:
            model = ChangeNetPlClassifier(experiment_config, dm)

        strategy = 'auto'

        if enable_tensorboard:
            ptl_loggers.append(
                TensorBoardLogger(
                    save_dir=results_dir
                )
            )
            infrequent_logging_frequency = experiment_config.train.tensorboard.infrequent_logging_frequency
            assert max(0, infrequent_logging_frequency) <= total_epochs, (
                f"infrequent_logging_frequency {infrequent_logging_frequency} must be < num_epochs {total_epochs}"
            )
            logging.info("Tensorboard logging enabled.")
        else:
            logging.info("Tensorboard logging disabled.")

    elif task == 'segment':
        assert enable_tensorboard is False, "Currently tensorboard visualization is not supported for Segmentation"

        dm = CNDataModule(experiment_config.dataset.segment)

        if pretrained_path:
            model = ChangeNetPlSegment.load_from_checkpoint(pretrained_path,
                                                            map_location="cpu",
                                                            experiment_spec=experiment_config
                                                            )
        else:
            model = ChangeNetPlSegment(experiment_config)

        strategy = 'auto'
        if len(gpus) > 1:
            strategy = 'ddp_find_unused_parameters_true'

    else:
        raise NotImplementedError('Only tasks supported by Visual ChangeNet are: "segment" and "classify"')

    trainer = Trainer(logger=ptl_loggers,
                      devices=gpus,
                      num_nodes=num_nodes,
                      max_epochs=total_epochs,
                      check_val_every_n_epoch=validation_interval,
                      default_root_dir=results_dir,
                      accelerator='gpu',
                      strategy=strategy,
                      precision=precision,
                      use_distributed_sampler=False,
                      sync_batchnorm=sync_batchnorm,
                      enable_checkpointing=False,
                      **trainer_kwargs
                      )

    trainer.fit(model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="Visual ChangeNet", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
