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

"""Inference on single patch."""

import os

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.metric_learning_recognition.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.metric_learning_recognition.inference.inferencer import Inferencer
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.cv.metric_learning_recognition.utils.decorators import monitor_status


def run_experiment(experiment_config):
    """Starts the inference.

    Args:
        experiment_config (DictConfig): Configuration dictionary

    """
    # no need to check `else` as it's verified in the decorator alreadya
    if experiment_config["inference"]["results_dir"]:
        results_dir = experiment_config["inference"]["results_dir"]
    elif experiment_config["results_dir"]:
        results_dir = os.path.join(experiment_config["results_dir"], "inference")

    inferencer = Inferencer(experiment_config, results_dir)
    inferencer.infer()


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="infer", schema=ExperimentConfig
)
@monitor_status(mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process.

    Args:
        cfg (DictConfig): Hydra config object.

    """
    # Obfuscate logs.
    obfuscate_logs(cfg)

    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
