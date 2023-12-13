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
import re

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.connectors.checkpoint_connector import TLTCheckpointConnector
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.utilities import update_results_dir
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner

from nvidia_tao_pytorch.cv.dino.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.dino.model.pl_dino_model import DINOPlModel

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import check_and_create, load_pretrained_weights


def run_experiment(experiment_config,
                   results_dir,
                   key):
    """Start the training."""
    # set the encryption key:

    TLTPyTorchCookbook.set_passphrase(key)
    dm = ODDataModule(experiment_config.dataset)

    # find_unuser_parameters=False and activation_checkpoint combination
    # requires every output in forward function to participate in
    # loss calculation. When return_interm_indices < 4, we must disable
    # activation checkpointing
    if experiment_config.train.activation_checkpoint and \
        len(experiment_config.model.return_interm_indices) < 4 and \
            experiment_config.train.num_gpus > 1:
        experiment_config.train.activation_checkpoint = False
        print("Disabling  activation checkpointing since model is smaller")

    activation_checkpoint = experiment_config.train.activation_checkpoint

    # Load pretrained model as starting point if pretrained path is provided,
    pretrained_path = experiment_config.train.pretrained_model_path
    if pretrained_path is not None:
        # Ignore backbone weights if we get pretrained path for the entire detector
        experiment_config.model.pretrained_backbone_path = None
        pt_model = DINOPlModel(experiment_config)
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
                print(f"skip layer: {k}, checkpoint layer size: {list(v.size())},",
                      f"current model layer size: {list(current_model_dict[k].size())}")
                new_checkpoint[k] = current_model_dict[k]
        # Load pretrained weights
        pt_model.model.load_state_dict(new_checkpoint, strict=False)
    else:
        pt_model = DINOPlModel(experiment_config)

    total_epochs = experiment_config.train.num_epochs

    check_and_create(results_dir)

    status_logger_callback = TAOStatusLogger(
        results_dir,
        append=True,
        num_epochs=total_epochs
    )

    status_logging.set_status_logger(status_logger_callback.logger)

    num_gpus = experiment_config.train.num_gpus
    num_nodes = experiment_config.train.num_nodes
    validation_interval = experiment_config.train.validation_interval
    ckpt_inter = experiment_config.train.checkpoint_interval

    assert ckpt_inter <= total_epochs, (
        f"Checkpoint interval {ckpt_inter} > Number of epochs {total_epochs}."
        f"Please set experiment_config.train.checkpoint_interval < {total_epochs}"
    )

    assert validation_interval <= total_epochs, (
        f"Validation interval {validation_interval} > Number of epochs {total_epochs}."
        f"Please set experiment_config.train.validation_interval < {total_epochs}"
    )

    clip_grad_norm = experiment_config.train.clip_grad_norm
    is_dry_run = experiment_config.train.is_dry_run
    distributed_strategy = experiment_config.train.distributed_strategy

    if experiment_config.train.precision.lower() in ["fp16", "fp32"]:
        precision = int(experiment_config.train.precision.replace("fp", ""))
    else:
        raise NotImplementedError(f"{experiment_config.train.precision} is not supported. Only fp32 and fp16 are supported")

    sync_batchnorm = False
    strategy = None
    if num_gpus > 1:
        # By default find_unused_parameters is set to True in Lightning for backward compatibility
        # This introduces extra overhead and can't work with activation checkpointing
        # Ref: https://pytorch-lightning.readthedocs.io/en/1.8.5/advanced/model_parallel.html#when-using-ddp-strategies-set-find-unused-parameters-false
        # TODO: Starting from PTL 2.0, find_usued_parameters is set to False by default
        if distributed_strategy.lower() == "ddp" and activation_checkpoint:
            strategy = DDPStrategy(find_unused_parameters=False)
        elif distributed_strategy.lower() == "ddp" and not activation_checkpoint:
            strategy = 'ddp'
        elif distributed_strategy.lower() == "ddp_sharded":
            strategy = 'ddp_sharded'
            # Override to FP16 for ddp_sharded as there's an error with FP32 during Positional Embedding forward pass
            print("Overriding Precision to FP16 for ddp_sharded")
            precision = 16
        else:
            raise NotImplementedError(f"{distributed_strategy} is not implemented. Only ddp and ddp_sharded are supported")

        if "fan" in experiment_config.model.backbone:
            print("Setting sync batch norm")
            sync_batchnorm = True

    trainer = Trainer(devices=num_gpus,
                      num_nodes=num_nodes,
                      max_epochs=total_epochs,
                      check_val_every_n_epoch=validation_interval,
                      default_root_dir=results_dir,
                      accelerator='gpu',
                      strategy=strategy,
                      precision=precision,
                      gradient_clip_val=clip_grad_norm,
                      replace_sampler_ddp=False,
                      sync_batchnorm=sync_batchnorm,
                      fast_dev_run=is_dry_run)

    # Overload connector to enable intermediate ckpt encryption & decryption.
    resume_ckpt = experiment_config.train.resume_training_checkpoint_path

    if resume_ckpt and resume_ckpt.endswith('.tlt'):
        if resume_ckpt is not None:
            trainer._checkpoint_connector = TLTCheckpointConnector(trainer, resume_from_checkpoint=resume_ckpt)
        else:
            trainer._checkpoint_connector = TLTCheckpointConnector(trainer)
        resume_ckpt = None

    # setup checkpointer:
    ModelCheckpoint.FILE_EXTENSION = ".pth"
    checkpoint_callback = ModelCheckpoint(every_n_epochs=ckpt_inter,
                                          dirpath=results_dir,
                                          save_on_train_epoch_end=True,
                                          monitor=None,
                                          save_top_k=-1,
                                          filename='dino_model_{epoch:03d}')
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
    trainer.fit(pt_model, dm, ckpt_path=resume_ckpt or None)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
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
