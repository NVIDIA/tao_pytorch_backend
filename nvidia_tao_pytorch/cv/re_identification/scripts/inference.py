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
import logging
import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.cv.re_identification.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.re_identification.dataloader.pl_reid_data_module import REIDDataModule
from nvidia_tao_pytorch.cv.re_identification.model.pl_reid_model import ReIdentificationModel


def run_experiment(experiment_config, key):
    """
    Start the inference process.

    This function initializes the necessary components for inference, including the model, data loader,
    and inferencer. It performs inference on the provided data and saves the results in the specified output file.

    Args:
        experiment_config (dict): The experiment configuration containing the model and inference parameters.
        results_dir (str): The directory to save the status and log files.
        key (str): The encryption key for intermediate checkpoints.

    Raises:
        Exception: If any error occurs during the inference process.
    """
    results_dir, model_path, gpus = initialize_inference_experiment(experiment_config, key)
    if len(gpus) > 1:
        gpus = [gpus[0]]
        logging.log(f"Re-Identification does not support multi-GPU inference at this time. Using only GPU {gpus}")

    dm = REIDDataModule(experiment_config)
    model = ReIdentificationModel.load_from_checkpoint(model_path,
                                                       map_location="cpu",
                                                       experiment_spec=experiment_config,
                                                       prepare_for_training=False)

    if "swin" in experiment_config.model.backbone:
        model.model.load_param(model_path)

    trainer = Trainer(devices=gpus,
                      default_root_dir=results_dir,
                      accelerator='gpu',
                      strategy='auto')

    trainer.predict(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="ReIdentification", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """
    Run the inference process.

    This function initializes the experiment and sets up logging. It calls run_experiment
    to perform inference on the data according to the experiment configuration, and handles
    any exceptions that occur during the process.

    Args:
        cfg (DictConfig): Configuration file.
    """
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
