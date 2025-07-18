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

"""Distill RT-DETR model."""

import os
from pytorch_lightning import LightningModule

from nvidia_tao_core.config.rtdetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.rtdetr.distillation.distiller import RtdetrDistiller
from nvidia_tao_pytorch.cv.rtdetr.scripts.train import run_experiment

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="distill", schema=ExperimentConfig
)
@monitor_status(name="RTDETR", mode="distill")
def main(cfg: ExperimentConfig) -> None:
    """Run the distillation process."""
    # This is for resuming to work without needing to save the teacher weights
    LightningModule.strict_loading = False

    run_experiment(experiment_config=cfg,
                   lightning_module=RtdetrDistiller)


if __name__ == "__main__":
    main()
