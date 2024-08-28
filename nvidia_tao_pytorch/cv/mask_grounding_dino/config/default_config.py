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

"""Default config file."""

from dataclasses import dataclass

from nvidia_tao_pytorch.config.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD
)

from nvidia_tao_pytorch.core.common_config import CommonExperimentConfig
from nvidia_tao_pytorch.cv.grounding_dino.config.default_config import (
    GDINODatasetConfig,
    GDINOModelConfig,
    GDINOTrainExpConfig,
    GDINOInferenceExpConfig,
    GDINOEvalExpConfig,
    GDINOExportExpConfig,
    GDINOGenTrtEngineExpConfig,
)


@dataclass
class MaskGDINODatasetConfig(GDINODatasetConfig):
    """Dataset config."""

    has_mask: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="has mask",
        description="Flag to load mask annotation from dataset."
    )


@dataclass
class MaskGDINOModelConfig(GDINOModelConfig):
    """DINO model config."""

    has_mask: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="has mask",
        description="Flag to enable mask head in grounding dino."
    )
    mask_loss_coef: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the mask error in the final loss.",
        display_name="Mask loss coefficient",
    )
    dice_loss_coef: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the dice loss of the segmentation in the final loss.",
        display_name="GIoU loss coefficient",
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: MaskGDINOModelConfig = DATACLASS_FIELD(
        MaskGDINOModelConfig(),
        description="Configurable parameters to construct the model for a Mask Grounding DINO experiment.",
    )
    dataset: MaskGDINODatasetConfig = DATACLASS_FIELD(
        MaskGDINODatasetConfig(),
        description="Configurable parameters to construct the dataset for a Mask Grounding DINO experiment.",
    )
    train: GDINOTrainExpConfig = DATACLASS_FIELD(
        GDINOTrainExpConfig(),
        description="Configurable parameters to construct the trainer for a Mask Grounding DINO experiment.",
    )
    evaluate: GDINOEvalExpConfig = DATACLASS_FIELD(
        GDINOEvalExpConfig(),
        description="Configurable parameters to construct the evaluator for a Mask Grounding DINO experiment.",
    )
    inference: GDINOInferenceExpConfig = DATACLASS_FIELD(
        GDINOInferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for a Mask Grounding DINO experiment.",
    )
    export: GDINOExportExpConfig = DATACLASS_FIELD(
        GDINOExportExpConfig(),
        description="Configurable parameters to construct the exporter for a Mask Grounding DINO experiment.",
    )
    gen_trt_engine: GDINOGenTrtEngineExpConfig = DATACLASS_FIELD(
        GDINOGenTrtEngineExpConfig(),
        description="Configurable parameters to construct the TensorRT engine builder for a Mask Grounding DINO experiment.",
    )
