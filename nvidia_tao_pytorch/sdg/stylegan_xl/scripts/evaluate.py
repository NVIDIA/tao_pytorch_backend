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

"""Evaluate StyleGAN XL model."""

import os
from pytorch_lightning import Trainer

from nvidia_tao_core.config.stylegan_xl.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.pl_sx_data_module import SXDataModule
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.pl_bg_data_module import BGDataModule
from nvidia_tao_pytorch.sdg.stylegan_xl.model.sx_pl_model import StyleganPlModel
from nvidia_tao_pytorch.sdg.stylegan_xl.model.bg_pl_model import BigdatasetganPlModel


def run_experiment(experiment_config, key):
    """Start the evaluation."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config)
    num_nodes = experiment_config.train.num_nodes

    # StyleGAN-XL only supports 'stylegan' and 'bigdatasetgan' tasks
    if experiment_config.task == 'stylegan':
        # build dataloader
        dm = SXDataModule(experiment_config.dataset)
        # build model and load from the given checkpoint
        model = StyleganPlModel.load_from_checkpoint(model_path,
                                                     map_location="cpu",
                                                     experiment_spec=experiment_config,
                                                     dm=dm
                                                     )

    elif experiment_config.task == 'bigdatasetgan':
        # build dataloader
        dm = BGDataModule(experiment_config.dataset)
        # build model and load from the given checkpoint
        model = BigdatasetganPlModel.load_from_checkpoint(model_path,
                                                          map_location="cpu",
                                                          experiment_spec=experiment_config,
                                                          dm=dm
                                                          )

    else:
        raise NotImplementedError("Task {} is not implemented".format(experiment_config.task))

    trainer = Trainer(**trainer_kwargs,
                      num_nodes=num_nodes,
                      # use_distributed_sampler=False,
                      )

    trainer.test(model, dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="StyleGAN-XL", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
