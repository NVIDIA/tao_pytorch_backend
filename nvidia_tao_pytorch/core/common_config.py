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

"""Common config fields across all models"""

from dataclasses import dataclass

from omegaconf import MISSING
from typing import Optional, List
from nvidia_tao_pytorch.config.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)


@dataclass
class CuDNNConfig:
    """Common CuDNN config"""

    benchmark: bool = BOOL_FIELD(value=False)
    deterministic: bool = BOOL_FIELD(value=True)


@dataclass
class TrainConfig:
    """Common train experiment config."""

    num_gpus: int = INT_FIELD(
        value=1,
        display_name="number of GPUs",
        description="""The number of GPUs to run the train job.""",
        valid_min=1,
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        display_name="GPU IDs",
        description="List of GPU IDs to run the training on. The length of this list must be equal to the number of gpus in train.num_gpus.")
    num_nodes: int = INT_FIELD(
        value=1,
        display_name="number of nodes",
        description="Number of nodes to run the training on. If > 1, then multi-node is enabled."
    )
    seed: int = INT_FIELD(
        value=1234,
        description="The seed for the initializer in PyTorch. If < 0, disable fixed seed.",
        display_name="seed",
        valid_min=-1,
        valid_max="inf",
        default_value=1234
    )
    cudnn: CuDNNConfig = DATACLASS_FIELD(CuDNNConfig())

    num_epochs: int = INT_FIELD(
        value=10,
        description="Number of epochs to run the training.",
        automl_enabled="TRUE",
        display_name="number of epochs",
        valid_min=1,
        valid_max="inf"
    )
    checkpoint_interval: int = INT_FIELD(
        value=1,
        display_name="Checkpoint interval",
        description="The interval (in epochs) at which a checkpoint will be saved. Helps resume training.",
        valid_min=1
    )
    validation_interval: int = INT_FIELD(
        value=1,
        display_name="Validation interval",
        description="The interval (in epochs) at which a evaluation will be triggered on the validation dataset.",
        valid_min=1
    )

    resume_training_checkpoint_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Path to the checkpoint to resume training from.",
        display_name="Resume checkpoint path."
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="Results directory",
        description="Path to where all the assets generated from a task are stored.")


@dataclass
class EvaluateConfig:
    """Common eval experiment config."""

    num_gpus: int = INT_FIELD(value=1)
    gpu_ids: List[int] = LIST_FIELD(arrList=[0])
    num_nodes: int = INT_FIELD(value=1)

    checkpoint: str = STR_FIELD(value=MISSING, default_value="")
    results_dir: Optional[str] = STR_FIELD(value=None, default_value="")


@dataclass
class InferenceConfig:
    """Common inference experiment config."""

    num_gpus: int = INT_FIELD(value=1)
    gpu_ids: List[int] = LIST_FIELD(arrList=[0])
    num_nodes: int = INT_FIELD(value=1)

    checkpoint: str = STR_FIELD(value=MISSING, default_value="")
    results_dir: Optional[str] = STR_FIELD(value=None, default_value="")


@dataclass
class WandBConfig:
    """Configuration element wandb client."""

    enable: bool = BOOL_FIELD(value=True)
    project: str = STR_FIELD(value="TAO Toolkit")
    entity: Optional[str] = STR_FIELD(value="")
    tags: List[str] = LIST_FIELD(arrList=["tao-toolkit"])
    reinit: bool = BOOL_FIELD(value=False)
    sync_tensorboard: bool = BOOL_FIELD(value=False)
    save_code: bool = BOOL_FIELD(value=False)
    name: str = BOOL_FIELD(value="TAO Toolkit Training")


@dataclass
class CommonExperimentConfig:
    """Common experiment config."""

    encryption_key: Optional[str] = STR_FIELD(value=None)
    results_dir: str = STR_FIELD(value='/results')
    wandb: WandBConfig = DATACLASS_FIELD(
        WandBConfig(
            project="TAO Toolkit",
            name="TAO Toolkit training experiment",
            tags=["training", "tao-toolkit"]
        )
    )
