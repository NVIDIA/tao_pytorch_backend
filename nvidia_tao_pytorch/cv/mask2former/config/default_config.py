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

from typing import Optional
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.config.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    INT_FIELD,
    STR_FIELD
)
from nvidia_tao_pytorch.core.common_config import (
    CommonExperimentConfig,
    EvaluateConfig,
    InferenceConfig)
from nvidia_tao_pytorch.cv.mask2former.config.dataset import Mask2FormerDatasetConfig
from nvidia_tao_pytorch.cv.mask2former.config.model import Mask2FormerModelConfig
from nvidia_tao_pytorch.cv.mask2former.config.train import Mask2FormerTrainExpConfig
from nvidia_tao_pytorch.cv.mask2former.config.deploy import Mask2FormerGenTrtEngineExpConfig


@dataclass
class Mask2FormerInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        description="""Path to the TensorRT engine to be used for inference.
                    This only works with :code:`tao-deploy`.""",
        display_name="tensorrt engine"
    )


@dataclass
class Mask2FormerEvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        description="""Path to the TensorRT engine to be used for evaluation.
                    This only works with :code:`tao-deploy`.""",
        display_name="tensorrt engine"
    )


@dataclass
class Mask2FormerExportExpConfig:
    """Evaluation experiment config."""

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

    model: Mask2FormerModelConfig = DATACLASS_FIELD(
        Mask2FormerModelConfig(),
        description="Configurable parameters to construct the model for a Mask2former experiment.",
    )
    dataset: Mask2FormerDatasetConfig = DATACLASS_FIELD(
        Mask2FormerDatasetConfig(),
        description="Configurable parameters to construct the dataset for a Mask2former experiment.",
    )
    train: Mask2FormerTrainExpConfig = DATACLASS_FIELD(
        Mask2FormerTrainExpConfig(),
        description="Configurable parameters to construct the trainer for a Mask2former experiment.",
    )
    inference: Mask2FormerInferenceExpConfig = DATACLASS_FIELD(
        Mask2FormerInferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for a Mask2former experiment.",
    )
    evaluate: Mask2FormerEvalExpConfig = DATACLASS_FIELD(
        Mask2FormerEvalExpConfig(),
        description="Configurable parameters to construct the evaluator for a Mask2former experiment.",
    )
    export: Mask2FormerExportExpConfig = DATACLASS_FIELD(
        Mask2FormerExportExpConfig(),
        description="Configurable parameters to construct the exporter for a Mask2former experiment.",
    )
    gen_trt_engine: Mask2FormerGenTrtEngineExpConfig = DATACLASS_FIELD(
        Mask2FormerGenTrtEngineExpConfig(),
        description="Configurable parameters to construct the TensorRT engine builder for a Mask2former experiment.",
    )
