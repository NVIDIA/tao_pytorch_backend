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

"""Evaluate a trained Grounding DINO model."""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import tempfile

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.cv.grounding_dino.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.grounding_dino.dataloader.pl_odvg_data_module import ODVGDataModule
from nvidia_tao_pytorch.cv.grounding_dino.model.pl_gdino_model import GDINOPlModel
from nvidia_tao_pytorch.cv.grounding_dino.utils.misc import parse_checkpoint


def run_experiment(experiment_config):
    """Run experiment."""
    results_dir, model_path, gpus = initialize_evaluation_experiment(experiment_config)

    if model_path.endswith('.pth'):
        # build dataloader
        dm = ODVGDataModule(experiment_config.dataset, subtask_config=experiment_config.evaluate)
        dm.setup(stage="test")
        cap_lists = dm.test_dataset.cap_lists

        # Check if the checkpoint is coming from TAO PTL trained or not
        original = torch.load(model_path, map_location="cpu")
        if "pytorch-lightning_version" not in original:
            # parse public checkpoint
            final = parse_checkpoint(original)
            tmp = tempfile.NamedTemporaryFile()
            model_path = tmp.name
            torch.save({"state_dict": final, 'pytorch-lightning_version': pl.__version__}, model_path)

        # build model and load from the given checkpoint
        model = GDINOPlModel.load_from_checkpoint(model_path,
                                                  map_location="cpu",
                                                  experiment_spec=experiment_config,
                                                  cap_lists=cap_lists)

        trainer = Trainer(devices=gpus,
                          default_root_dir=results_dir,
                          accelerator='gpu',
                          strategy='auto')

        trainer.test(model, datamodule=dm)

    elif model_path.endswith('.engine'):
        raise NotImplementedError("TensorRT evaluation is supported through tao-deploy. "
                                  "Please use tao-deploy to generate TensorRT enigne and run evaluation.")
    else:
        raise NotImplementedError("Model path format is only supported for .pth")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name="Grounding DINO", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the evaluate process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
