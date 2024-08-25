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

"""Train DINO model."""

import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.connectors.checkpoint_connector import TLTCheckpointConnector
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.cv.dino.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.dino.model.pl_dino_model import DINOPlModel
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import load_pretrained_weights


def run_experiment(experiment_config,
                   key,
                   lightning_module=DINOPlModel):
    """Start the training."""
    results_dir, resume_ckpt, gpus, ptl_loggers = initialize_train_experiment(experiment_config, key)

    dm = ODDataModule(experiment_config.dataset)
    dm.setup(stage="fit")

    # find_unuser_parameters=False and activation_checkpoint combination
    # requires every output in forward function to participate in
    # loss calculation. When return_interm_indices < 4, we must disable
    # activation checkpointing
    if experiment_config.train.activation_checkpoint and \
        len(experiment_config.model.return_interm_indices) < 4 and \
            experiment_config.train.num_gpus > 1:
        experiment_config.train.activation_checkpoint = False
        logging.info("Disabling  activation checkpointing since model is smaller")

    activation_checkpoint = experiment_config.train.activation_checkpoint

    # Load pretrained model as starting point if pretrained path is provided,
    pretrained_path = experiment_config.train.pretrained_model_path
    if pretrained_path:
        # Ignore backbone weights if we get pretrained path for the entire detector
        experiment_config.model.pretrained_backbone_path = None
        pt_model = lightning_module(experiment_config)
        current_model_dict = pt_model.model.state_dict()
        checkpoint = load_pretrained_weights(pretrained_path)
        new_checkpoint = {}
        for k, k_ckpt in zip(sorted(current_model_dict.keys()), sorted(checkpoint.keys())):
            v = checkpoint[k_ckpt]
            # Handle PTL format
            k = k.replace("model.model.", "model.")
            if v.size() == current_model_dict[k].size():
                new_checkpoint[k] = v
            else:
                # Skip layers that mismatch
                logging.info(f"skip layer: {k}, checkpoint layer size: {list(v.size())},",
                             f"current model layer size: {list(current_model_dict[k].size())}")
                new_checkpoint[k] = current_model_dict[k]
        # Load pretrained weights
        pt_model.model.load_state_dict(new_checkpoint, strict=False)
    else:
        pt_model = lightning_module(experiment_config)

    total_epochs = experiment_config.train.num_epochs
    num_nodes = experiment_config.train.num_nodes
    validation_interval = experiment_config.train.validation_interval
    clip_grad_norm = experiment_config.train.clip_grad_norm
    is_dry_run = experiment_config.train.is_dry_run
    distributed_strategy = experiment_config.train.distributed_strategy

    if experiment_config.train.precision.lower() == 'fp16':
        precision = '16-mixed'
    elif experiment_config.train.precision.lower() == 'fp32':
        precision = '32-true'
    else:
        raise NotImplementedError(f"{experiment_config.train.precision} is not supported. Only fp32 and fp16 are supported")

    sync_batchnorm = False
    strategy = 'auto'
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
            logging.info("Overriding Precision to FP16 for fsdp")
            precision = '16-mixed'
        else:
            raise NotImplementedError(f"{distributed_strategy} is not implemented. Only ddp and fsdp are supported")

        if "fan" in experiment_config.model.backbone:
            logging.info("Setting sync batch norm")
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
                      use_distributed_sampler=False,
                      sync_batchnorm=sync_batchnorm,
                      enable_checkpointing=False,
                      fast_dev_run=is_dry_run)

    # Overload connector to enable intermediate ckpt encryption & decryption.
    if resume_ckpt and resume_ckpt.endswith('.tlt'):
        trainer._checkpoint_connector = TLTCheckpointConnector(trainer)

    trainer.fit(pt_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
@monitor_status(name="DINO", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key,
                   lightning_module=DINOPlModel)


if __name__ == "__main__":
    main()
