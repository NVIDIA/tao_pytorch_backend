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

"""Train BEVFusion model."""

import os
import logging
from mmengine.config import Config
from mmengine.runner import Runner

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner

# Triggers build of custom modules
from nvidia_tao_core.config.bevfusion.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.bevfusion.utils.config import BEVFusionConfig
from nvidia_tao_pytorch.cv.bevfusion.visualization import TAO3DLocalVisualizer # noqa pylint: disable=W0401, W0611
from nvidia_tao_pytorch.cv.bevfusion.evaluation import TAO3DMetric # noqa pylint: disable=W0401, W0611,
from nvidia_tao_pytorch.cv.bevfusion.model import *  # noqa pylint: disable=W0401, W0614, W0611
from nvidia_tao_pytorch.cv.bevfusion.datasets import *  # noqa pylint: disable=W0401, W0614, W0611
from nvidia_tao_pytorch.cv.bevfusion.utils.logger import TAOBEVFusionLoggerHook  # noqa pylint: disable=W0401, W0614, W0611


def run_experiment(experiment_config):
    """Start the training."""
    # update cfg to be compatible with mmdet3d
    bev_config = BEVFusionConfig(experiment_config, phase="train")
    train_cfg = bev_config.updated_config
    train_cfg = Config(train_cfg)
    # build runner
    runner = Runner.from_cfg(train_cfg)
    # run training
    runner.train()


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
@monitor_status(name="Bevfusion", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # numba logging supression
    nb_logger = logging.getLogger('numba')
    nb_logger.setLevel(logging.ERROR)  # only show error

    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
