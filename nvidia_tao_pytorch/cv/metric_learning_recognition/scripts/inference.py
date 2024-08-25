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

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.cv.metric_learning_recognition.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.metric_learning_recognition.dataloader.pl_ml_data_module import MLDataModule
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.cv.metric_learning_recognition.model.pl_ml_recog_model import MLRecogModel


def run_experiment(experiment_config):
    """Starts the inference.

    Args:
        experiment_config (DictConfig): Configuration dictionary

    """
    results_dir, model_path, gpus = initialize_inference_experiment(experiment_config)

    dm = MLDataModule(experiment_config)
    dm.setup(stage='predict')

    metric_learning_recognition = MLRecogModel.load_from_checkpoint(
        model_path,
        map_location="cpu",
        experiment_spec=experiment_config,
        results_dir=results_dir,
        dm=dm,
        subtask="inference")

    trainer = Trainer(devices=gpus,
                      default_root_dir=results_dir,
                      accelerator='gpu',
                      strategy='auto')

    trainer.predict(metric_learning_recognition, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="infer", schema=ExperimentConfig
)
@monitor_status(name="Metric Learning Recognition", mode="inference")
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
