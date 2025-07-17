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
#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
"""Inference module."""
import os
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nvidia_tao_core.config.ocdnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.cv.ocdnet.data_loader.pl_ocd_data_module import OCDDataModule
from nvidia_tao_pytorch.cv.ocdnet.model.pl_ocd_model import OCDnetModel
from nvidia_tao_pytorch.cv.ocdnet.utils.util import load_checkpoint
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs

import pycuda
import pycuda.autoinit
pyc_dev = pycuda.autoinit.device
pyc_ctx = pyc_dev.retain_primary_context()


def run_experiment(experiment_config):
    """Run experiment."""
    experiment_config = OmegaConf.to_container(experiment_config)
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config)

    experiment_config['model']['pretrained'] = False

    checkpoint = load_checkpoint(model_path, to_cpu=True)
    dm = OCDDataModule(experiment_config)
    dm.setup(stage='predict')
    model = OCDnetModel(experiment_config, dm, 'predict')
    model.model.load_state_dict(checkpoint)

    trainer = Trainer(**trainer_kwargs)

    trainer.predict(model, datamodule=dm)

    pyc_ctx.pop()


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="inference", schema=ExperimentConfig
)
@monitor_status(name="OCDNet", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)

    pyc_ctx.push()

    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
