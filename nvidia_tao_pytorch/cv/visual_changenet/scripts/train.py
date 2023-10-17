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
import re

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.visual_changenet.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.visual_changenet.utils.common_utils import check_and_create
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.dataloader.changenet_dm import CNDataModule
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.core.utilities import update_results_dir
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlSegment
from nvidia_tao_pytorch.cv.visual_changenet.classification.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlClassifier
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utilities import get_last_generated_file

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

CHECKPOINT_FILE_EXT = "pth"


# TODO: @zbhat modify this to get model with best val accuracy for evaluation
def get_latest_tlt_model(results_dir):
    """Utility function to return the latest tlt model in a dir."""
    trainable_ckpts = [int(item.split('.')[0].split('_')[1]) for item in os.listdir(results_dir)
                       if item.endswith(".tlt")]
    num_ckpts = len(trainable_ckpts)
    if num_ckpts == 0:
        return None
    latest_step = sorted(trainable_ckpts, reverse=True)[0]
    latest_checkpoint = expand_path(os.path.join(results_dir, "iter_{}.tlt".format(latest_step)))
    if not os.path.isfile(latest_checkpoint):
        raise FileNotFoundError("Checkpoint file not found at {}")
    return latest_checkpoint


def run_experiment(experiment_config, key, results_dir):
    """Start the training."""
    TLTPyTorchCookbook.set_passphrase(key)

    task = experiment_config.task
    check_and_create(results_dir)
    num_gpus = experiment_config["num_gpus"]
    num_nodes = experiment_config.train.num_nodes
    total_epochs = experiment_config.train.num_epochs
    validation_interval = experiment_config.train.val_interval
    checkpoint_interval = experiment_config.train.checkpoint_interval
    enable_tensorboard = experiment_config.train.tensorboard.enabled

    status_logger_callback = TAOStatusLogger(
        results_dir,
        append=True,
        num_epochs=total_epochs
    )
    status_logging.set_status_logger(status_logger_callback.logger)

    # Load pretrained model as starting point if pretrained path is provided
    pretrained_path = experiment_config.train.pretrained_model_path

    precision = 32
    sync_batchnorm = False
    trainer_kwargs = {}
    checkpoint_callback_kwargs = {}

    assert task in ['segment', 'classify'], "Visual ChangeNet only supports 'segment' and 'classify' tasks."
    if task == 'classify':
        assert checkpoint_interval <= total_epochs, (
            f"Checkpoint interval {checkpoint_interval} > Number of epochs {total_epochs}."
            f"Please set experiment_config.train.checkpoint_interval < {total_epochs}"
        )
        assert validation_interval <= total_epochs, (
            f"Validation interval {validation_interval} > Number of epochs {total_epochs}."
            f"Please set experiment_config.train.validation_interval < {total_epochs}"
        )

        if pretrained_path is not None:
            model = ChangeNetPlClassifier.load_from_checkpoint(pretrained_path,
                                                               map_location="cpu",
                                                               experiment_spec=experiment_config)
        else:
            model = ChangeNetPlClassifier(experiment_config)

        if enable_tensorboard:
            trainer_kwargs["logger"] = TensorBoardLogger(
                save_dir=results_dir
            )
            infrequent_logging_frequency = experiment_config.train.tensorboard.infrequent_logging_frequency
            assert max(0, infrequent_logging_frequency) <= total_epochs, (
                f"infrequent_logging_frequency {infrequent_logging_frequency} must be < num_epochs {total_epochs}"
            )
            logging.info("Tensorboard logging enabled.")
        else:
            logging.info("Tensorboard logging disabled.")

        # checkpoint kwargs
        checkpoint_callback_kwargs['filename'] = 'changenet_classifier_{epoch:03d}'  # -{val_acc:.4f}
        checkpoint_callback_kwargs['save_on_train_epoch_end'] = True  # checkpointing at train end
        checkpoint_callback_kwargs['monitor'] = None  # quantity to monitor for ep saving
        checkpoint_callback_kwargs['save_top_k'] = -1  # save all checkpoints after ckpt_inter - else if k means save best k models according to monitor quantity

    elif task == 'segment':
        assert enable_tensorboard is False, "Currently tensorboard visualization is not supported for Segmentation"

        dm = CNDataModule(experiment_config.dataset.segment)

        if pretrained_path is not None:
            model = ChangeNetPlSegment.load_from_checkpoint(pretrained_path,
                                                            map_location="cpu",
                                                            experiment_spec=experiment_config
                                                            )
        else:
            model = ChangeNetPlSegment(experiment_config)

        checkpoint_callback_kwargs['filename'] = 'changenet_model_segment_{val_acc:.4f}-{epoch:03d}'
        checkpoint_callback_kwargs['save_on_train_epoch_end'] = False  # checkpointing at validation end
        checkpoint_callback_kwargs['monitor'] = 'val_acc'  # quantity to monitor for ep saving
        checkpoint_callback_kwargs['save_top_k'] = total_epochs  # save all checkpoints after ckpt_inter - else if k means save best k models according to monitor quantity
        checkpoint_callback_kwargs['mode'] = 'max'
    else:
        raise NotImplementedError('Only tasks supported by Visual ChangeNet are: "segment" and "classify"')

    acc_flag = None
    if num_gpus > 1:
        acc_flag = "ddp"

    trainer = Trainer(devices=num_gpus,
                      num_nodes=num_nodes,
                      max_epochs=total_epochs,
                      check_val_every_n_epoch=validation_interval,
                      default_root_dir=results_dir,
                      accelerator='gpu',
                      strategy=acc_flag,
                      precision=precision,
                      replace_sampler_ddp=False,
                      sync_batchnorm=sync_batchnorm,
                      **trainer_kwargs
                      )

    # Overload connector to enable intermediate ckpt encryption & decryption.
    if experiment_config['train']['resume_training_checkpoint_path']:
        resume_ckpt = experiment_config['train']['resume_training_checkpoint_path']
    else:
        # Get the latest checkpoint file to resume training from by default.
        resume_ckpt = get_last_generated_file(
            results_dir,
            extension=CHECKPOINT_FILE_EXT
        )
        logging.info("Setting resume checkpoint to {}".format(resume_ckpt))

    logging.info(
        "Results directory {} Checkpoint Interval {}".format(results_dir, checkpoint_interval)
    )

    ckpt_inter = experiment_config.train.checkpoint_interval

    # setup checkpointer:
    ModelCheckpoint.FILE_EXTENSION = ".pth"
    checkpoint_callback = ModelCheckpoint(every_n_epochs=ckpt_inter,
                                          dirpath=results_dir,
                                          **checkpoint_callback_kwargs
                                          )
    trainer.callbacks.append(checkpoint_callback)

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

    if task == 'classify':
        trainer.fit(model, ckpt_path=resume_ckpt or None)
    elif task == 'segment':
        trainer.fit(model, dm, ckpt_path=resume_ckpt or None)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        cfg = update_results_dir(cfg, task="train")
        # Obfuscate logs.
        obfuscate_logs(cfg)
        run_experiment(experiment_config=cfg,
                       key=cfg.encryption_key,
                       results_dir=cfg.results_dir)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Training finished successfully"
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Train was interrupted",
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
