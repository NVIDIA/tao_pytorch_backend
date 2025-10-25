# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Evaluate a trained depthnet model."""

import os
import torch
from pytorch_lightning import Trainer

from nvidia_tao_core.config.depth_net.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.cv.depth_net.dataloader import build_pl_data_module
from nvidia_tao_pytorch.cv.depth_net.utils.misc import parse_mono_depth_checkpoint
from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import build_pl_model, get_pl_module


def run_experiment(experiment_config, key):
    """Run experiment."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config, key)

    if model_path.endswith('.tlt') or model_path.endswith('.pth'):
        # build data module
        dm = build_pl_data_module(experiment_config.dataset)
        dm.setup(stage="test")

        model_dict = torch.load(model_path, map_location="cpu")

        if "pytorch-lightning_version" not in model_dict:
            # parse public checkpoint
            if experiment_config.model.model_type in ['MetricDepthAnything', 'RelativeDepthAnything']:
                model_dict = parse_mono_depth_checkpoint(model_dict, experiment_config.model.model_type)
            model = build_pl_model(experiment_config)
            model.load_state_dict(model_dict, strict=True)
        else:
            model = get_pl_module(experiment_config).load_from_checkpoint(
                model_path,
                map_location="cpu",
                experiment_spec=experiment_config
            )
        trainer = Trainer(**trainer_kwargs)
        trainer.test(model, datamodule=dm)

    elif model_path.endswith('.engine'):
        raise NotImplementedError("TensorRT evaluation is supported through tao-deploy. "
                                  "Please use tao-deploy to generate TensorRT engine and run evaluation.")
    else:
        raise NotImplementedError("Model path format is only supported for .tlt or .pth")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name="Depth Net", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the evaluate process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
