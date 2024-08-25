# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Train Mask2former model."""

import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner

from nvidia_tao_pytorch.cv.mask2former.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.mask2former.model.pl_model import Mask2formerPlModule

from nvidia_tao_pytorch.cv.mask2former.dataloader.pl_data_module import SemSegmDataModule


def run_experiment(experiment_config):
    """Start the training."""
    results_dir, resume_ckpt, gpus, ptl_loggers = initialize_train_experiment(experiment_config)

    pl_data = SemSegmDataModule(experiment_config.dataset)
    if not experiment_config.train.iters_per_epoch:
        experiment_config.train.iters_per_epoch = len(pl_data.train_dataloader())
    pl_model = Mask2formerPlModule(experiment_config)

    # load pretrained weights
    if not resume_ckpt:
        pl_model.load_pretrained_weights(pl_model.experiment_spec.train.pretrained_model_path)
        pl_model.load_backbone_weights(pl_model.experiment_spec.model.backbone.pretrained_weights)

    total_epochs = experiment_config.train.num_epochs
    num_nodes = experiment_config.train.num_nodes
    validation_interval = experiment_config.train.validation_interval

    clip_grad_norm = experiment_config.train.clip_grad_norm
    is_dry_run = experiment_config.train.is_dry_run
    distributed_strategy = experiment_config.train.distributed_strategy

    if experiment_config.train.precision.lower() == "fp16":
        precision = '16-mixed'
    elif experiment_config.train.precision.lower() == 'fp32':
        precision = '32-true'
    else:
        raise NotImplementedError(f"{experiment_config.train.precision} is not supported. Only fp32 and fp16 are supported")

    sync_batchnorm = False
    strategy = 'auto'
    activation_checkpoint = experiment_config.train.activation_checkpoint
    if len(gpus) > 1:
        # By default find_unused_parameters is set to False in Lightning
        # If true, it introduces extra overhead and can't work with activation checkpointing
        if distributed_strategy.lower() == "ddp" and activation_checkpoint:
            strategy = 'ddp'
        elif distributed_strategy.lower() == "ddp" and not activation_checkpoint:
            strategy = 'ddp_find_unused_parameters_true'
        elif distributed_strategy.lower() == "fsdp":
            strategy = 'fsdp'
            # Override to FP16 for fsdp as there's an error with FP32 during Positional Embedding forward pass
            print("Overriding Precision to FP16 for fsdp")
            precision = '16-mixed'
        else:
            raise NotImplementedError(f"{distributed_strategy} is not implemented. Only ddp and fsdp are supported")

        if "fan" in experiment_config.model.backbone:
            print("Setting sync batch norm")
            sync_batchnorm = True

    trainer = Trainer(logger=ptl_loggers,
                      devices=gpus,
                      num_nodes=num_nodes,
                      max_epochs=total_epochs,
                      check_val_every_n_epoch=validation_interval,
                      default_root_dir=results_dir,
                      accelerator='gpu',
                      strategy=strategy,
                      precision=precision,
                      gradient_clip_val=clip_grad_norm,
                      sync_batchnorm=sync_batchnorm,
                      enable_checkpointing=False,
                      fast_dev_run=is_dry_run)

    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))
    trainer.fit(pl_model, pl_data, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec", schema=ExperimentConfig
)
@monitor_status(name="Mask2Former", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
