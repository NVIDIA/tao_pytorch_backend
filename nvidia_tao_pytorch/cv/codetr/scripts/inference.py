# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native inference with CoDETR PyTorch checkpoint."""

import os

import torch
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.codetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.codetr.model.pl_codetr_model import CoDETRPlModel
from nvidia_tao_pytorch.cv.codetr.model.utils import load_codetr_state_dict


def build_and_load_codetr_model(experiment_config, model_path):
    """Build a CoDETRPlModel and load weights from a .pth/.tlt checkpoint.

    Handles both TAO-format and reference Co-DETR checkpoints (auto-detected),
    drops EMA keys, pads ``label_enc`` for the background token, and skips
    size-mismatched keys (e.g. differing collab head conv shapes).

    Args:
        experiment_config: Hydra experiment spec (with model/dataset/etc).
        model_path (str): path to a ``.pth`` or ``.tlt`` checkpoint file.

    Returns:
        CoDETRPlModel: model with checkpoint weights loaded (still on CPU).

    Raises:
        NotImplementedError: if ``model_path`` is not a ``.pth`` or ``.tlt`` file.
    """
    if not (model_path.endswith('.tlt') or model_path.endswith('.pth')):
        if model_path.endswith('.engine'):
            raise NotImplementedError(
                "TensorRT inference is supported through tao-deploy. "
                "Please use tao-deploy to generate TensorRT engine and run inference."
            )
        raise NotImplementedError("Model path format is only supported for .tlt or .pth")

    model = CoDETRPlModel(experiment_spec=experiment_config)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    load_codetr_state_dict(model, state_dict, dec_layers=experiment_config.model.dec_layers)
    return model


def run_experiment(experiment_config, key):
    """Execute CoDETR inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config, key)

    # start_from_one=False: the reference Co-DETR checkpoint uses 0-indexed class labels
    dm = ODDataModule(experiment_config.dataset, subtask_config=experiment_config.inference,
                      start_from_one=False)
    dm.setup(stage="predict")

    model = build_and_load_codetr_model(experiment_config, model_path)
    trainer = Trainer(**trainer_kwargs)
    trainer.predict(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="inference", schema=ExperimentConfig
)
@monitor_status(name="CoDETR", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run CoDETR inference."""
    run_experiment(experiment_config=cfg, key=cfg.encryption_key)


if __name__ == "__main__":
    main()
