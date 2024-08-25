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

"""Train Optical Inspection Siamese Network model."""
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.optical_inspection.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule
from nvidia_tao_pytorch.cv.optical_inspection.model.pl_oi_model import OpticalInspectionModel


def run_experiment(experiment_config, key):
    """Start the training."""
    results_dir, resume_ckpt, gpus, ptl_loggers = initialize_train_experiment(experiment_config, key)

    dm = OIDataModule(experiment_config)

    # Load pretrained model as starting point if pretrained path is provided,
    pretrained_path = experiment_config.train.pretrained_model_path
    if pretrained_path:
        oi_model = OpticalInspectionModel.load_from_checkpoint(pretrained_path,
                                                               map_location="cpu",
                                                               experiment_spec=experiment_config,
                                                               dm=dm)
    else:
        oi_model = OpticalInspectionModel(experiment_config, dm)

    total_epochs = experiment_config['train']['num_epochs']
    clip_grad = experiment_config['train']['clip_grad_norm']
    validation_interval = experiment_config.train.validation_interval
    enable_tensorboard = experiment_config.train.tensorboard.enabled

    trainer_kwargs = {}
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
    acc_flag = 'auto'
    trainer = Trainer(
        devices=gpus,
        max_epochs=total_epochs,
        check_val_every_n_epoch=validation_interval,
        default_root_dir=results_dir,
        accelerator='gpu',
        strategy=acc_flag,
        gradient_clip_val=clip_grad,
        enable_checkpointing=False,
        **trainer_kwargs
    )

    trainer.fit(oi_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="Optical Inspection", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
