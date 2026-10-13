# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate CoDETR model."""

import os

import torch
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.codetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.codetr.model.pl_codetr_model import CoDETRPlModel
from nvidia_tao_pytorch.cv.codetr.model.utils import load_codetr_state_dict


def run_experiment(experiment_config, key):
    """Execute CoDETR evaluation."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config, key)

    dm = ODDataModule(experiment_config.dataset, subtask_config=experiment_config.evaluate,
                      start_from_one=False)
    dm.setup(stage="test")

    if model_path.endswith('.tlt') or model_path.endswith('.pth'):
        # Build model and load checkpoint (TAO or reference Co-DETR layout)
        model = CoDETRPlModel(experiment_spec=experiment_config)
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        load_codetr_state_dict(model, state_dict, dec_layers=experiment_config.model.dec_layers)

    trainer = Trainer(**trainer_kwargs)
    trainer.test(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="eval", schema=ExperimentConfig
)
@monitor_status(name="CoDETR", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run CoDETR evaluation."""
    run_experiment(experiment_config=cfg, key=cfg.encryption_key)


if __name__ == "__main__":
    main()
