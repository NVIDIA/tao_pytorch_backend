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

    cal_image_dir: List[str] = LIST_FIELD(
        arrList=MISSING,
        display_name="calibration image directories",
        description="""List of image directories to be used for calibration
                    when running Post Training Quantization using TensorRT.""",
    )
    cal_cache_file: str = STR_FIELD(
        value=MISSING,
        display_name="calibration cache file",
        description="""The path to save the calibration cache file containing
                    scales that were generated during Post Training Quantization.""",
    )
    cal_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        description="""The batch size of the input TensorRT to run calibration on.""",
        display_name="min batch size"
    )
    cal_batches: int = INT_FIELD(
        value=1,
        default_value=1,
        description="""The number of input tensor batches to run calibration on.
                    It is recommended to use atleast 10% of the training images.""",
        display_name="number of calibration batches"
    )


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = STR_FIELD(
        value="FP32",
        default_value="FP32",
        description="The precision to be set for building the TensorRT engine.",
        display_name="data type",
        valid_options=",".join(["FP32", "FP16"])
    )
    workspace_size: int = INT_FIELD(
        value=1024,
        default_value=1024,
        description="""The size (in MB) of the workspace TensorRT has
                    to run it's optimization tactics and generate the
                    TensorRT engine.""",
        display_name="max workspace size"
    )
    min_batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        description="""The minimum batch size in the optimization profile for
                    the input tensor of the TensorRT engine.""",
        display_name="min batch size"
    )
    opt_batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        description="""The optimum batch size in the optimization profile for
                    the input tensor of the TensorRT engine.""",
        display_name="optimum batch size"
    )
    max_batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        description="""The maximum batch size in the optimization profile for
                    the input tensor of the TensorRT engine.""",
        display_name="maximum batch size"
    )
    calibration: CalibrationConfig = DATACLASS_FIELD(
        CalibrationConfig(),
        description="""The configuration elements to define the
                    TensorRT calibrator for int8 PTQ.""",
    )


@dataclass
class DDGenTrtEngineExpConfig:
    """Gen TRT Engine experiment config."""

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
        description="""Path to the TensorRT engine generated should be stored.
                    This only works with :code:`tao-deploy`.""",
        display_name="tensorrt engine"
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
    tensorrt: TrtConfig = DATACLASS_FIELD(
        TrtConfig(),
        description="Hyper parameters to configure the TensorRT Engine builder.",
        display_name="TensorRT hyper params."
    )
