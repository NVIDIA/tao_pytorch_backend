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

""" Inference on single patch. """
import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.cv.deformable_detr.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.deformable_detr.model.pl_dd_model import DeformableDETRModel


def run_experiment(experiment_config, key):
    """Start the inference."""
    results_dir, model_path, gpus = initialize_inference_experiment(experiment_config, key)
    if model_path.endswith('.tlt') or model_path.endswith('.pth'):

        # build data module
        dm = ODDataModule(experiment_config.dataset, subtask_config=experiment_config.inference)
        dm.setup(stage="predict")

        # Run inference using tlt model
        model = DeformableDETRModel.load_from_checkpoint(model_path,
                                                         map_location="cpu",
                                                         experiment_spec=experiment_config)

        trainer = Trainer(devices=gpus,
                          default_root_dir=results_dir,
                          accelerator='gpu',
                          strategy='auto')

        trainer.predict(model, datamodule=dm)

    elif model_path.endswith('.engine'):
        raise NotImplementedError("TensorRT inference is supported through tao-deploy. "
                                  "Please use tao-deploy to generate TensorRT enigne and run inference.")
    else:
        raise NotImplementedError("Model path format is only supported for .tlt or .pth")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="infer", schema=ExperimentConfig
)
@monitor_status(name="Deformable DETR", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
