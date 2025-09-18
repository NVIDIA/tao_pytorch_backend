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

"""NVDINOv2 Train Script"""

import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_core.config.nvdinov2.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.pl_dinov2_data_module import DinoV2DataModule
from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import DinoV2PlModel


def run_experiment(experiment_config, key):
    """Start the training."""
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    num_nodes = experiment_config.train.num_nodes

    # Load pretrained model as starting point if pretrained path is provided
    pretrained_path = experiment_config.train.pretrained_model_path

    precision = experiment_config.train.precision

    dm = DinoV2DataModule(experiment_config)

    model = DinoV2PlModel(experiment_config)

    if pretrained_path:
        model.pretrained_weights = pretrained_path
        model.restore_pretrained_weights()

    trainer = Trainer(**trainer_kwargs,
                      num_nodes=num_nodes,
                      strategy='auto',
                      precision=precision,
                      use_distributed_sampler=False,
                      sync_batchnorm=True,
                      )

    trainer.fit(model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="NVDINOv2", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
