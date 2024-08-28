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

"""Default config file."""

from typing import Optional, List, Dict
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.config.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD
)
from nvidia_tao_pytorch.core.common_config import (
    CommonExperimentConfig,
    EvaluateConfig,
    InferenceConfig
)
from nvidia_tao_pytorch.core.distillation.config import (
    DistillationBindingConfig,
    DistillationConfig
)
from nvidia_tao_pytorch.cv.dino.config.dataset import (
    DINODatasetConfig
)
from nvidia_tao_pytorch.cv.dino.config.deploy import DINOGenTrtEngineExpConfig
from nvidia_tao_pytorch.cv.dino.config.model import DINOModelConfig
from nvidia_tao_pytorch.cv.dino.config.train import DINOTrainExpConfig


@dataclass
class DINOModelDistillationBindingConfig(DistillationBindingConfig):
    """Distillation binding config."""

    pass


@dataclass
class DINODistillationConfig(DistillationConfig):
    """Distillation config"""

    teacher: DINOModelConfig = DATACLASS_FIELD(
        DINOModelConfig(),
        descripton="Configuration hyper parameters for the DINO based teacher model.",
        display_name="teacher"
    )
    pretrained_teacher_model_path: Optional[str] = STR_FIELD(
        value=MISSING,
        display_name="Pretrained teacher model path",
        description="Path to the pre-trained teacher model."
    )
    bindings: List[DINOModelDistillationBindingConfig] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="List of bindings for Distillation. Each element is an instance of DINOModelDistillationBindingConfig.",
        display_name="bindings"
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """
    )


@dataclass
class DINOInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        description="""Path to the TensorRT engine to be used for evaluation.
                    This only works with :code:`tao-deploy`.""",
        display_name="tensorrt engine"
    )
    color_map: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        description="Class-wise dictionary with colors to render boxes.",
        display_name="color map"
    )
    conf_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="""The value of the confidence threshold to be used when
                    filtering out the final list of boxes.""",
        display_name="confidence threshold"
    )
    is_internal: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="is internal",
        description="Flag to render with internal directory structure."
    )
    input_width: Optional[int] = INT_FIELD(
        value=None,
        default_value=960,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=32,
    )
    input_height: Optional[int] = INT_FIELD(
        value=None,
        default_value=544,
        description="Height of the input image tensor.",
        display_name="input height",
        valid_min=32,
    )
    outline_width: int = INT_FIELD(
        value=3,
        default_value=3,
        description="Width in pixels of the bounding box outline.",
        display_name="outline width",
        valid_min=1,
    )


@dataclass
class DINOEvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    input_width: Optional[int] = INT_FIELD(
        value=None,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=1,
    )
    input_height: Optional[int] = INT_FIELD(
        value=None,
        description="Height of the input image tensor.",
        display_name="input height",
        valid_min=1,
    )
    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        description="""Path to the TensorRT engine to be used for evaluation.
                    This only works with :code:`tao-deploy`.""",
        display_name="tensorrt engine"
    )
    conf_threshold: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="""The value of the confidence threshold to be used when
                    filtering out the final list of boxes.""",
        display_name="confidence threshold"
    )


@dataclass
class DINOExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="""The index of the GPU to build the TensorRT engine.""",
        display_name="GPU ID"
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to the checkpoint file to run export.",
        display_name="checkpoint"
    )
    onnx_file: str = STR_FIELD(
        value=MISSING,
        default_value="",
        display_name="onnx file",
        description="""
        Path to the onnx model file.
        """
    )
    on_cpu: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="verbose",
        description="""Flag to export CPU compatible model."""
    )
    input_channel: int = INT_FIELD(
        value=3,
        default_value=3,
        description="Number of channels in the input Tensor.",
        display_name="input channel",
        valid_min=3,
    )
    input_width: int = INT_FIELD(
        value=960,
        default_value=960,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=32,
    )
    input_height: int = INT_FIELD(
        value=544,
        default_value=544,
        description="Height of the input image tensor.",
        display_name="input height",
        valid_min=32,
    )
    opset_version: int = INT_FIELD(
        value=17,
        default_value=17,
        description="""Operator set version of the ONNX model used to generate
                    the TensorRT engine.""",
        display_name="opset version",
        valid_min=1,
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        description="""The batch size of the input Tensor for the engine.
                    A value of :code:`-1` implies dynamic tensor shapes.""",
        display_name="batch size"
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="verbose",
        description="""Flag to enable verbose TensorRT logging."""
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: DINOModelConfig = DATACLASS_FIELD(
        DINOModelConfig(),
        description="Configurable parameters to construct the model for a DINO experiment.",
    )
    dataset: DINODatasetConfig = DATACLASS_FIELD(
        DINODatasetConfig(),
        description="Configurable parameters to construct the dataset for a DINO experiment.",
    )
    train: DINOTrainExpConfig = DATACLASS_FIELD(
        DINOTrainExpConfig(),
        description="Configurable parameters to construct the trainer for a DINO experiment.",
    )
    evaluate: DINOEvalExpConfig = DATACLASS_FIELD(
        DINOEvalExpConfig(),
        description="Configurable parameters to construct the evaluator for a DINO experiment.",
    )
    inference: DINOInferenceExpConfig = DATACLASS_FIELD(
        DINOInferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for a DINO experiment.",
    )
    export: DINOExportExpConfig = DATACLASS_FIELD(
        DINOExportExpConfig(),
        description="Configurable parameters to construct the exporter for a DINO experiment.",
    )
    gen_trt_engine: DINOGenTrtEngineExpConfig = DATACLASS_FIELD(
        DINOGenTrtEngineExpConfig(),
        description="Configurable parameters to construct the TensorRT engine builder for a DINO experiment.",
    )
    distill: Optional[DINODistillationConfig] = DATACLASS_FIELD(
        None,
        description="Configurable parameters to construct the distiller for a DINO experiment.",
    )
