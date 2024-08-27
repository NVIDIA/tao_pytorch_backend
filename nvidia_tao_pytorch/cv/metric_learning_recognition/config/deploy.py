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

"""Configuration hyperparameter schema to deploy the model."""

from dataclasses import dataclass
from typing import List, Optional
from omegaconf import MISSING

from nvidia_tao_pytorch.config.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    INT_FIELD,
    STR_FIELD,
    LIST_FIELD
)


@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_cache_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="cal_cache_file",
    )
    cal_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The calibration batch size",
        display_name="calibration batch size",
    )
    cal_batches: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The number of data batches sent to calibration",
        display_name="number of calibration batches",
    )
    cal_image_dir: Optional[List[str]] = LIST_FIELD(
        arrList=[],
        default_value=[],
        display_name="calibration image directories",
        description="""[Optional] List of image directories to be used for calibration
                    when running Post Training Quantization using TensorRT.""",
    )


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = STR_FIELD(
        value="FP32",
        default_value="FP32",
        display_name="data_type",
        description="[Optional] The precision to be used for the TensorRT engine.",
        valid_options="FP32, FP16, INT8",
    )
    workspace_size: int = INT_FIELD(
        value=1024,
        default_value=1024,
        description="workspace size for TensorRT engine generation in MB",
        display_name="workspace_size",
    )
    min_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The mininum batch size of TensorRT engine generation",
        display_name="minimum batch size",
    )
    opt_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The optimal batch size of TensorRT engine generation",
        display_name="optimal batch size",
    )
    max_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The maximum batch size of TensorRT engine generation",
        display_name="maximum batch size",
    )
    calibration: CalibrationConfig = DATACLASS_FIELD(
        CalibrationConfig(),
        description="The calibration configuration for the model.",
        display_name="calibration",
    )


@dataclass
class MLGenTrtEngineExpConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="Results directory",
        description="""
        [Optional] Path to where train assets generated from a task are stored.
        """
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="""The GPU ID for trt engine generation. Currently, trt engine generation is
                    only supported on a single GPU.""",
        display_name="gpu id",
    )
    onnx_file: str = STR_FIELD(
        value=MISSING,
        default_value="",
        display_name="onnx file",
        description="""
        Path to the onnx model file.
        """
    )
    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="trt engine",
        description="""
        [Optional] The path to the TensorRT (TRT) engine to be evaluated. Currently, only trt_engine
        is supported in TAO Deploy."""
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        description="""The batch size of the exported ONNX model. If batch_size is -1, the exported
                    ONNX model has a dynamic batch size.""",
        display_name="batch size",
    )
    verbose: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="verbose",
        description="If True, prints out the TensorRT engine conversion.",
    )
    tensorrt: TrtConfig = DATACLASS_FIELD(
        TrtConfig(),
        description="The TensorRT configuration for the model.",
        display_name="tensorrt",
    )
