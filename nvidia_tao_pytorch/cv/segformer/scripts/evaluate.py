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
Evaluation of Segformer model.
"""
import os

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.segformer.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer.utils.config import MMSegmentationConfig

# Triggers build of custom modules
from nvidia_tao_pytorch.cv.segformer.model import * # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.segformer.dataloader import * # noqa pylint: disable=W0401, W0614

from mmengine.runner import Runner
from mmengine.config import Config


def run_experiment(experiment_config):
    """Start the evaluate.
    Args:
        experiment_config (Dict): Config dictionary containing epxeriment parameters
        results_dir (str): Results dir to save the evaluation result

    """
    results_dir, model_path, _ = initialize_evaluation_experiment(experiment_config)

    mmseg_config = MMSegmentationConfig(experiment_config, phase="evaluate")
    eval_cfg = mmseg_config.updated_config

    eval_cfg["work_dir"] = results_dir
    # This is provided in notebook
    eval_cfg["load_from"] = model_path

    eval_cfg = Config(eval_cfg)
    runner = Runner.from_cfg(eval_cfg)

    # start testing
    metrics = runner.test()
    print(metrics)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="test_isbi", schema=ExperimentConfig
)
@monitor_status(name="Segformer", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
