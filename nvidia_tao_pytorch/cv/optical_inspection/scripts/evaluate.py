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

"""Evaluate a trained Optical Inspection model."""

import os
import logging
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_core.config.optical_inspection.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule
from nvidia_tao_pytorch.cv.optical_inspection.model.pl_oi_model import OpticalInspectionModel

logger = logging.getLogger(__name__)


def run_experiment(experiment_config, key):
    """Run experiment."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config, key)
    if len(trainer_kwargs['devices']) > 1:
        trainer_kwargs['devices'] = [trainer_kwargs['devices'][0]]
        logger.info(f"Optical Inspection does not support multi-GPU evaluation at this time. Using only GPU {trainer_kwargs['devices']}")

    dm = OIDataModule(experiment_config)

    model = OpticalInspectionModel.load_from_checkpoint(
        model_path,
        map_location="cpu",
        experiment_spec=experiment_config,
        dm=dm
    )

    trainer = Trainer(**trainer_kwargs)

    trainer.test(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="Optical Inspection", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the Evaluate process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
