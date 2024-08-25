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

""" MMClassification Evaluate Module """

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.core.mmlab.mmclassification.classification_default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.classification.heads import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.models import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.metrics import DumpResultsScores  # noqa pylint: disable=W0401, W0614
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.mmlab.mmclassification.utils import MMPretrainConfig

from mmengine.runner import Runner

import os


def run_experiment(experiment_config):
    """Start evaluation."""
    results_dir, model_path, _ = initialize_evaluation_experiment(experiment_config)
    status_logger = status_logging.get_status_logger()
    mmpretrain_config = MMPretrainConfig(experiment_config, phase="evaluate")
    eval_cfg = mmpretrain_config.updated_config
    eval_cfg["work_dir"] = results_dir
    eval_cfg["load_from"] = model_path
    eval_cfg["train_dataloader"] = None
    runner = Runner.from_cfg(eval_cfg)
    classes = sorted(list(runner.val_dataloader.dataset.CLASSES))
    eval_file_path = os.path.join(results_dir, "results.csv")
    runner.test_evaluator.metrics.append(DumpResultsScores(out_file_path=eval_file_path, classes=classes))
    status_logger.write(message="********************** Start logging for Evaluation **********************.")
    # start testing
    metrics = runner.test()
    print(metrics)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="test_cats_and_dogs", schema=ExperimentConfig
)
@monitor_status(name="Classification", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the Evaluate process."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
