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

from typing import Optional
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.core.common_config import EvaluateConfig, CommonExperimentConfig, InferenceConfig
from nvidia_tao_pytorch.config.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    INT_FIELD,
    STR_FIELD,
)
from nvidia_tao_pytorch.cv.metric_learning_recognition.config.deploy import MLGenTrtEngineExpConfig
from nvidia_tao_pytorch.cv.metric_learning_recognition.config.model import MLModelConfig
from nvidia_tao_pytorch.cv.metric_learning_recognition.config.train import MLTrainExpConfig
from nvidia_tao_pytorch.cv.metric_learning_recognition.config.dataset import MLDatasetConfig


@dataclass
class MLEvalExpConfig(EvaluateConfig):
    """Evaluation experiment configuration template."""

    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        description="""[Optional] Path to the TensorRT engine generated should be stored.
                    This only works with :code:`tao-deploy`.""",
        display_name="tensorrt engine"
    )
    topk: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Select the mode of top k closest objects as match at evaluation",
        display_name="topk",
    )
    batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        description="The evaluation batch size",
        display_name="batch size",
    )
    report_accuracy_per_class: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to report accuracy per class at valiation or not.",
        display_name="report accuracy per class"
    )


@dataclass
class MLInferenceExpConfig(InferenceConfig):
    """Inference experiment configuration template."""

    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="trt_engine",
        description="""[Optional] The path to the TensorRT (TRT) engine to run inference.
                    Currently, only trt_engine is supported in TAO Deploy.""",
    )
    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="checkpoint",
        description="[Optional] The path to the .pth torch model to run inference.",
    )
    input_path: str = STR_FIELD(
        value=MISSING,
        default_value="",
        display_name="input_path",
        description="The path to the data to run inference on",
    )
    inference_input_type: str = STR_FIELD(
        value="classification_folder",
        default_value="classification_folder",
        display_name="inference input type",
        description="Inference input format",
        valid_options="image, image_folder, classification_folder",
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The inference batch size",
        display_name="batch size",
    )
    topk: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Select the mode of top k closest objects as match at inference",
        display_name="topk",
    )


@dataclass
class MLExportExpConfig:
    """Export experiment configuraiton template."""

    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        description="The export batch size. -1 as dynamic batch size",
        display_name="batch size",
    )
    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="checkpoint",
        description="[Optional] The path to the .pth torch model to export.",
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="The GPU ID for export. Currently, export is only supported on a single GPU.",
        display_name="gpu id",
    )
    onnx_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="onnx_file",
        description="""[Optional] The path to the exported ONNX file. If this value is not
                    specified, it defaults to model.onnx in export.results_dir""",
    )
    on_cpu: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="on cpu",
        description="If True, the Torch-to-ONNX export will be performed on CPU",
    )
    opset_version: int = INT_FIELD(
        value=14,
        default_value=14,
        description="The version of the default (ai.onnx) opset to target.",
        display_name="opset version",
    )
    verbose: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="verbose",
        description="If True, prints a description of the model being exported to stdout.",
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="Results directory",
        description="""
        [Optional] Path to where export assets generated from a task are stored.
        """
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    train: MLTrainExpConfig = DATACLASS_FIELD(
        MLTrainExpConfig(),
        description="The training configuration for the model.",
        display_bane="train",
    )
    model: MLModelConfig = DATACLASS_FIELD(
        MLModelConfig(),
        description="The model configuration for the experiment.",
        display_name="model",
    )
    evaluate: MLEvalExpConfig = DATACLASS_FIELD(
        MLEvalExpConfig(),
        description="The evaluation configuration for the model.",
        display_name="evaluate",
    )
    dataset: MLDatasetConfig = DATACLASS_FIELD(
        MLDatasetConfig(),
        description="The dataset configuration for the experiment.",
        display_name="dataset",
    )
    export: MLExportExpConfig = DATACLASS_FIELD(
        MLExportExpConfig(),
        description="The export configuration for the model.",
        display_name="export",
    )
    gen_trt_engine: MLGenTrtEngineExpConfig = DATACLASS_FIELD(
        MLGenTrtEngineExpConfig(),
        description="The TensorRT engine generation configuration for the model.",
        display_name="gen_trt_engine",
    )
    inference: MLInferenceExpConfig = DATACLASS_FIELD(
        MLInferenceExpConfig(),
        description="The inference configuration for the model.",
        display_name="inference",
    )
