# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ensure validation and evaluation consume the selected matching protocol."""

from omegaconf import OmegaConf
import pytest

from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.sparse4d.dataloader import pl_sparse4d_data_module


pytestmark = pytest.mark.cv_unit


@pytest.mark.parametrize("stage", ["fit", "test", "predict"])
@pytest.mark.parametrize("protocol", [None, "center_distance", "iou_3d", "both"])
def test_datamodule_propagates_evaluation_protocol(monkeypatch, stage, protocol):
    """Legacy specs without evaluation fields and explicit overrides reach datasets."""
    config = OmegaConf.structured(ExperimentConfig())
    config.dataset.data_root = "/data"
    for name in ("train_dataset", "val_dataset", "test_dataset"):
        config.dataset[name].ann_file = f"/data/{name}.pkl"
    if protocol is None:
        # Model bundles from before the schema exposed these fields omit them.
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        del config.dataset.eval_dist_fcn
        del config.dataset.eval_hota
    else:
        config.dataset.eval_dist_fcn = protocol
        config.dataset.eval_hota = True
    observed = []

    def dataset_stub(**kwargs):
        observed.append(kwargs)
        return [None]

    monkeypatch.setattr(
        pl_sparse4d_data_module, "Omniverse3DDetTrackDataset", dataset_stub
    )
    module = pl_sparse4d_data_module.Sparse4DDataModule(config)
    module.setup(stage)
    evaluation = [kwargs for kwargs in observed if kwargs["test_mode"]]
    assert len(evaluation) == 1
    assert evaluation[0]["eval_dist_fcn"] == (protocol or "iou_3d")
    assert evaluation[0]["eval_hota"] is (protocol is not None)
