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

"""Default config file"""

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
    STR_FIELD,
)
from nvidia_tao_pytorch.core.common_config import EvaluateConfig, CommonExperimentConfig, InferenceConfig, TrainConfig


@dataclass
class ModelConfig:
    """Optical recognition model config."""

    model_type: str = STR_FIELD(value="Siamese_3", default_value="Siamese_3", description="Model Architecture type", valid_options="Siamese,Siamese_3", automl_enabled="TRUE")
    margin: float = FLOAT_FIELD(value=2.0, default_value=2.0, valid_min=1.0, valid_max="inf", automl_enabled="TRUE")
    model_backbone: str = STR_FIELD(value="custom", default_value="custom", description="Model backbone type")
    embedding_vectors: int = INT_FIELD(value=5, default_value=5, valid_min=1, valid_max="inf", automl_enabled="TRUE")
    imagenet_pretrained: bool = BOOL_FIELD(value=False, default_value=False, description="flag to use imagenet_pretrained backbone weights")


@dataclass
class OptimConfig:
    """Optimizer config."""

    type: str = STR_FIELD(value="Adam", default_value="Adam", description="Optimizer")
    lr: float = FLOAT_FIELD(value=5e-4, default_value=0.0005, valid_min=0, valid_max="inf", automl_enabled="TRUE", description="Optimizer learning rate")
    momentum: float = FLOAT_FIELD(value=0.9, default_value=0.9, math_cond="> 0.0", display_name="momentum - AdamW", description="The momentum for the AdamW optimizer.", automl_enabled="TRUE")
    weight_decay: float = FLOAT_FIELD(value=5e-4, default_value=0.0005, valid_min=0, valid_max="inf", display_name="weight decay", description="The weight decay coefficient.")


@dataclass
class RandomFlip:
    """RandomFlip augmentation config."""

    vflip_probability: float = FLOAT_FIELD(value=0.5, default_value=0.5, valid_min=0, valid_max=1, description="Vertical Flip probability", automl_enabled="TRUE")
    hflip_probability: float = FLOAT_FIELD(value=0.5, default_value=0.5, valid_min=0, valid_max=1, description="Horizontal Flip probability", automl_enabled="TRUE")
    enable: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable augmentation", automl_enabled="TRUE")


@dataclass
class RandomRotation:
    """RandomRotation augmentation config."""

    rotate_probability: float = FLOAT_FIELD(value=0.5, default_value=0.5, valid_min=0, valid_max=1, description="Random Rotate probability", automl_enabled="TRUE")
    angle_list: List[float] = LIST_FIELD(arrList=[90, 180, 270], default_value=[90, 180, 270], description="Random rotate angle probability")
    enable: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable augmentation", automl_enabled="TRUE")


@dataclass
class RandomColor:
    """RandomColor augmentation config."""

    brightness: float = FLOAT_FIELD(value=0.3, default_value=0.3, math_cond="> 0.0", description="Random Color Brightness", automl_enabled="TRUE")
    contrast: float = FLOAT_FIELD(value=0.3, default_value=0.3, math_cond="> 0.0", description="Random Color Contrast", automl_enabled="TRUE")
    saturation: float = FLOAT_FIELD(value=0.3, default_value=0.3, math_cond="> 0.0", description="Random Color Saturation", automl_enabled="TRUE")
    hue: float = FLOAT_FIELD(value=0.3, default_value=0.3, math_cond="> 0.0", description="Random Color Hue", automl_enabled="TRUE")
    enable: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable Random Color", automl_enabled="TRUE")
    color_probability: float = FLOAT_FIELD(value=0.5, default_value=0.5, valid_min=0, valid_max=1, description="Random Color Probability", automl_enabled="TRUE")


@dataclass
class AugmentationConfig:
    """Augmentation config."""

    rgb_input_mean: List[float] = LIST_FIELD(arrList=[0.485, 0.456, 0.406], default_value=[0.485, 0.456, 0.406], description="Mean for the augmentation", display_name="Mean")
    rgb_input_std: List[float] = LIST_FIELD([0.229, 0.224, 0.225], default_value=[0.226, 0.226, 0.226], description="Standard deviation for the augmentation", display_name="Standard Deviation")
    random_flip: RandomFlip = DATACLASS_FIELD(RandomFlip())
    random_rotate: RandomRotation = DATACLASS_FIELD(RandomRotation())
    random_color: RandomColor = DATACLASS_FIELD(RandomColor())
    with_random_blur: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable with_random_blur")
    with_random_crop: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable with_random_crop")
    augment: bool = BOOL_FIELD(value=False, default_value=False, description="Flag to enable augmentation", automl_enabled="TRUE")


@dataclass
class DataPathFormat:
    """Dataset Path experiment config."""

    csv_path: str = STR_FIELD(value=MISSING, default_value="", description="Path to csv file for dataset")
    images_dir: str = STR_FIELD(value=MISSING, default_value="", description="Path to images directory for dataset")


@dataclass
class DatasetConfig:
    """Dataset config."""

    train_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    validation_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    test_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    infer_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    image_ext: Optional[str] = STR_FIELD(value=None, default_value=".jpg", description="Image extension")
    batch_size: int = INT_FIELD(value=32, default_value=8, valid_min=1, valid_max="inf", description="Batch size", display_name="Batch Size", automl_enabled="TRUE")
    workers: int = INT_FIELD(value=8, default_value=1, valid_min=0, valid_max="inf", description="Workers", display_name="Workers", automl_enabled="TRUE")
    fpratio_sampling: float = FLOAT_FIELD(value=0.1, default_value=0.1, valid_min=0.0, valid_max=1.0, description="Sampling ratio for minority class", automl_enabled="TRUE")
    num_input: int = INT_FIELD(value=8, default_value=4, valid_min=1, valid_max="inf", description="Number of input lighting conditions")
    input_map: Optional[Dict[str, int]] = DICT_FIELD(None, default_value={"LowAngleLight": 0, "SolderLight": 1, "UniformLight": 2, "WhiteLight": 3}, description="input mapping")
    grid_map: Optional[Dict[str, int]] = DICT_FIELD(None, default_value={"x": 2, "y": 2}, description="grid map")
    concat_type: Optional[str] = STR_FIELD(value=None, default_value="linear", valid_options="linear,grid", description="concat type")
    image_width: int = INT_FIELD(value=128, default_value=128, description="Width of the input image tensor.")
    image_height: int = INT_FIELD(value=128, default_value=128, description="Height of the input image tensor.")
    augmentation_config: AugmentationConfig = DATACLASS_FIELD(AugmentationConfig())


@dataclass
class TensorBoardLogger:
    """Configuration for the tensorboard logger."""

    enabled: bool = BOOL_FIELD(value=False, default_value=False, description="Flag to enable tensorboard")
    infrequent_logging_frequency: int = INT_FIELD(value=2, default_value=2, valid_min=0, valid_max="inf", description="infrequent_logging_frequency")  # Defined per epoch


@dataclass
class TrainExpConfig(TrainConfig):
    """Train experiment config."""

    optim: OptimConfig = DATACLASS_FIELD(OptimConfig())
    loss: Optional[str] = STR_FIELD(value=None, default_value="contrastive", description="ChangeNet Classify loss")
    clip_grad_norm: float = FLOAT_FIELD(value=0.0, default_value=1, valid_min=0.0, valid_max="inf", description="Gradient clipping", display_name="Gradient clipping")
    tensorboard: Optional[TensorBoardLogger] = DATACLASS_FIELD(TensorBoardLogger())
    pretrained_model_path: Optional[str] = STR_FIELD(value=None, default_value="", description="Pretrained model path", display_name="pretrained model path")


@dataclass
class InferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    trt_engine: str = STR_FIELD(value=MISSING, default_value="", description="TRT engine", display_name="TRT engine")
    batch_size: int = INT_FIELD(value=1, default_value=1, valid_min=1, valid_max="inf", description="Batch size", display_name="Batch Size")


@dataclass
class EvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    trt_engine: str = STR_FIELD(value=MISSING, default_value="", description="TRT engine", display_name="TRT engine")
    batch_size: int = INT_FIELD(value=-1, default_value=8, valid_min=1, valid_max="inf", description="Batch size", display_name="Batch Size")


@dataclass
class ExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Results directory", display_name="Results directory")
    gpu_id: int = INT_FIELD(value=0, default_value=0, description="GPU ID", display_name="GPU ID", value_min=0)
    checkpoint: str = STR_FIELD(value=MISSING, default_value="", description="Path to checkpoint file", display_name="Path to checkpoint file")
    onnx_file: Optional[str] = STR_FIELD(value=MISSING, default_value="", description="ONNX file", display_name="ONNX file")
    opset_version: int = INT_FIELD(value=17, default_value=12, valid_min=1, display_name="opset version", description="""Operator set version of the ONNX model used to generate the TensorRT engine.""")
    on_cpu: bool = BOOL_FIELD(value=False, description="Flag to export on cpu", display_name="On CPU")
    input_height: int = INT_FIELD(value=512, default_value=512, description="Input height", display_name="Input height")
    input_width: int = INT_FIELD(value=128, default_value=128, description="Input width", display_name="Input width")
    input_channel: int = INT_FIELD(value=3, default_value=3, description="Input channel", display_name="Input channel")
    batch_size: int = INT_FIELD(value=-1, default_value=-1,  description="Batch size")
    do_constant_folding: bool = BOOL_FIELD(value=False, default_value=False, description="Flag to do constant folding")


@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_image_dir: List[str] = LIST_FIELD(arrList=MISSING, default_value="", display_name="calibration image directories", description="""List of image directories to be used for calibration when running Post Training Quantization using TensorRT.""")
    cal_cache_file: str = STR_FIELD(value=MISSING, default_value="", display_name="calibration cache file", description="""The path to save the calibration cache file containing scales that were generated during Post Training Quantization.""")
    cal_batch_size: int = INT_FIELD(value=1, default_value=1, description="""The batch size of the input TensorRT to run calibration on.""", display_name="min batch size")
    cal_batches: int = INT_FIELD(value=1, default_value=1, description="""The number of input tensor batches to run calibration on. It is recommended to use atleast 10% of the training images.""", display_name="number of calibration batches")


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = STR_FIELD(value="fp16", default_value="fp16", description="Data type", display_name="Data type")
    workspace_size: int = INT_FIELD(value=1024, default_value=1024, description="Workspace size", display_name="Workspace size")
    min_batch_size: int = INT_FIELD(value=1, default_value=1, description="Minimum batch size", display_name="Minimum batch size")
    opt_batch_size: int = INT_FIELD(value=1, default_value=4, description="Optimum batch size", display_name="Optimum batch size")
    max_batch_size: int = INT_FIELD(value=1, default_value=16, description="Maximum batch size", display_name="Maximum batch size")
    calibration: CalibrationConfig = DATACLASS_FIELD(CalibrationConfig())


@dataclass
class GenTrtEngineExpConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Results directory", display_name="Results directory")
    gpu_id: int = INT_FIELD(value=0, default_value=0, description="GPU ID", display_name="GPU ID", value_min=0)
    onnx_file: Optional[str] = STR_FIELD(value=MISSING, default_value="", description="ONNX file", display_name="ONNX file")
    opset_version: int = INT_FIELD(value=17, default_value=12, valid_min=1, display_name="opset version", description="""Operator set version of the ONNX model used to generate the TensorRT engine.""")
    input_height: int = INT_FIELD(value=400, default_value=400, description="Input height", display_name="Input height")
    input_width: int = INT_FIELD(value=400, default_value=400, description="Input width", display_name="Input width")
    input_channel: int = INT_FIELD(value=3, default_value=3, description="Input channel", display_name="Input channel")
    batch_size: int = INT_FIELD(value=-1, default_value=-1,  description="Batch size")
    trt_engine: Optional[str] = STR_FIELD(value=None, description="TRT engine", display_name="TRT engine")
    verbose: bool = BOOL_FIELD(value=False, default_value=False, description="Verbose", display_name="Verbose")
    tensorrt: TrtConfig = DATACLASS_FIELD(TrtConfig())


@dataclass
class DatasetConvertConfig:
    """Dataset Convert experiment config."""

    root_dataset_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Dataset root directory", display_name="Root directory")
    data_convert_output_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Convert dataset directory", display_name="Convert directory")
    train_pcb_dataset_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Train directory", display_name="Train directory")
    val_pcb_dataset_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Validation directory", display_name="Validation directory")
    all_pcb_dataset_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Full dataset directory", display_name="Dataset directory")
    golden_csv_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Golden csv directory", display_name="Golden directory")
    project_name: Optional[str] = STR_FIELD(value=None, description="Project name", display_name="Project name")
    bot_top: Optional[str] = STR_FIELD(value=None)


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: ModelConfig = DATACLASS_FIELD(ModelConfig())
    dataset: DatasetConfig = DATACLASS_FIELD(DatasetConfig())
    train: TrainExpConfig = DATACLASS_FIELD(TrainExpConfig())
    evaluate: EvalExpConfig = DATACLASS_FIELD(EvalExpConfig())
    export: ExportExpConfig = DATACLASS_FIELD(ExportExpConfig())
    inference: InferenceExpConfig = DATACLASS_FIELD(InferenceExpConfig())
    dataset_convert: DatasetConvertConfig = DATACLASS_FIELD(DatasetConvertConfig())
    gen_trt_engine: GenTrtEngineExpConfig = DATACLASS_FIELD(GenTrtEngineExpConfig())
