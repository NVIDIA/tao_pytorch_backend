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

from typing import Optional, List
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.core.common_config import EvaluateConfig, CommonExperimentConfig, InferenceConfig, TrainConfig
from nvidia_tao_pytorch.pruning.prune_config import PruneConfig
from nvidia_tao_pytorch.config.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    FLOAT_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class OCRNetModelConfig:
    """OCRNet model config."""

    TPS: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="The bool flag to apply TPS to the model.",
        display_name="TPS"
    )  # Thin-Plate-Spline interpolation
    num_fiducial: int = INT_FIELD(
        value=20,
        default_value=3,
        valid_min=1,
        valid_max="inf",
        description="The number of fiducial points for TPS.",
        display_name="num fiducial"
    )  # number of keypoints for TPS
    backbone: str = STR_FIELD(
        value="ResNet",
        value_type="ordered",
        default_value="ResNet",
        valid_options="ResNet,ResNet2X,FAN_tiny_2X",
        description="Backbone of the model.",
        display_name="backbone"
    )  # [ResNet]
    feature_channel: int = INT_FIELD(
        value=512,
        default_value=512,
        valid_min=1,
        valid_max="inf",
        description="The number of backbone's feature output channel.",
        display_name="feature channel"
    )
    sequence: str = STR_FIELD(
        value="BiLSTM",
        value_type="ordered",
        default_value="BiLSTM",
        valid_options="BiLSTM",
        description="The sequence fature modeling type.",
        display_name="sequence"
    )  # [BiLSTM]
    hidden_size: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="The number of hidden uints in BiLSTM layers.",
        display_name="hidden_size"
    )
    prediction: str = STR_FIELD(
        value="CTC",
        value_type="ordered",
        default_value="CTC",
        valid_options="CTC,Attn",
        description="The sequence decoding method.",
        display_name="prediction"
    )  # [Attn, CTC]
    quantize: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="The bool flag to apply quantization to the model.",
        display_name="quantize"
    )
    input_width: int = INT_FIELD(
        value=100,
        default_value=100,
        valid_min=1,
        valid_max="inf",
        description="The input width of the model.",
        display_name="input width"
    )
    input_height: int = INT_FIELD(
        value=32,
        default_value=32,
        valid_min=1,
        valid_max="inf",
        description="The input height of the model.",
        display_name="input height"
    )
    input_channel: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="The input channel of the model.",
        display_name="input channel"
    )
    pruned_graph_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The pruned model to be loaded.",
        display_name="pruned graph path"
    )
    quantize_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The quantized model to be loaded.",
        display_name="quantize model path"
    )


@dataclass
class OptimConfig:
    """Optimizer config."""

    name: str = STR_FIELD(
        value="adadelta",
        value_type="ordered",
        default_value="adadelta",
        valid_options="adadelta,adam",
        description="Name of the optimizer.",
        display_name="name"
    )  # [adam, adadelta]
    lr: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        valid_max="inf",
        automl_enabled="TRUE",
        description="Learning rate.",
        display_name="lr"
    )  # default value = 1.0 for adadelta
    momentum: float = FLOAT_FIELD(
        value=0.9,
        default_value=0.9,
        valid_min=0.0,
        valid_max=1.0,
        description="Momentum coefficient.",
        display="momentum"
    )
    weight_decay: float = FLOAT_FIELD(
        value=5e-4,
        default_value=5e-4,
        valid_min=0,
        valid_max="inf",
        description="Weight decay coefficient for trainng.",
        display="weight decay"
    )
    lr_scheduler: str = STR_FIELD(
        value="MultiStep",
        value_type="ordered",
        default_value="MultiStep",
        valid_options="MultiStep,AutoReduce",
        description="Learning rate scheduler.",
        display="lr scheduler"
    )  # {AutoReduce, MultiStep}
    lr_monitor: str = STR_FIELD(
        value="val_loss",
        value_type="ordered",
        default_value="val_loss",
        valid_options="val_loss,train_loss",
        description="Learning rate monitor for AutoReduce learning rate scheduler.",
        display="lr monitor"
    )  # {val_loss, train_loss}
    patience: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Number of epochs for AutoReduce learning rate scheduler tolerance.",
        display="patience"
    )
    min_lr: float = FLOAT_FIELD(
        value=1e-4,
        default_value=1e-4,
        valid_min=0,
        valid_max="inf",
        description="Minimum learning rate.",
        display="min lr"
    )
    lr_steps: List[int] = LIST_FIELD(
        arrList=[15, 25],
        description="Steps to change learning rate in MultiStep scheduler.",
        display="lr steps"
    )
    lr_decay: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        description="Learning rate decay factor in learning rate scheduler.",
        display="lr decay"
    )


# TODO(tylerz): no augmentation from original implementation
@dataclass
class OCRNetAugmentationConfig:
    """Augmentation config."""

    keep_aspect_ratio: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="The bool flag to keep aspect ratio in resizing input images.",
        display="keep aspect ratio"
    )
    aug_prob: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        description="The probability to augment the input images.",
        display="aug prob"
    )
    reverse_color_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        description="The probability to reverse the color of input images.",
        display="reverse color prob"
    )
    rotate_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        description="The probability to rotate the input images.",
        display="rotate prob"
    )
    max_rotation_degree: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=0,
        valid_max=360,
        description="The maximum rotation degree.",
        display="max rotation degree"
    )
    blur_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        description="The probability to blur the input images.",
        display="blur prob"
    )
    gaussian_radius_list: Optional[List[int]] = LIST_FIELD(
        arrList=[1, 2, 3, 4],
        default_value=[1, 2, 3, 4],
        description="The gaussian raidus list for gaussian blur.",
        display="gaussian radius list"
    )


@dataclass
class OCRNetDatasetConfig:
    """Dataset config."""

    train_dataset_dir: Optional[List[str]] = LIST_FIELD(
        arrList=None,
        default_value="",
        description="The absolute path to the train dataset directory.",
        display="train dataset dir"
    )
    train_gt_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to the train dataset ground truth file.",
        display="train gt file"
    )
    val_dataset_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to the validation dataset directory.",
        display="val dataset dir"
    )
    val_gt_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to the validation dataset ground truth file.",
        display="val gt file"
    )
    character_list_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to the character list.",
        display="character list file"
    )
    max_label_length: int = INT_FIELD(
        value=25,
        default_value=25,
        description="The maximum length of the labels.",
        display="max label length"
    )  # Shall we check it with output feature length ?
    batch_size: int = INT_FIELD(
        value=32,
        default_value=32,
        description="Batch size of model input.",
        display="batch size"
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=8,
        description="The number of workers to process the dataset.",
        display="workers"
    )
    augmentation: OCRNetAugmentationConfig = DATACLASS_FIELD(
        OCRNetAugmentationConfig(),
        description="Configurable parameters for augmentation.",
    )


@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_image_dir: List[str] = LIST_FIELD(arrList=MISSING, default_value="")
    cal_cache_file: str = STR_FIELD(value=MISSING)
    cal_batch_size: int = INT_FIELD(value=1)
    cal_batches: int = INT_FIELD(value=1)


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = STR_FIELD(value="fp16")
    workspace_size: int = INT_FIELD(value=1024)
    min_batch_size: int = INT_FIELD(value=1)
    opt_batch_size: int = INT_FIELD(value=1)
    max_batch_size: int = INT_FIELD(value=1)
    calibration: CalibrationConfig = DATACLASS_FIELD(CalibrationConfig())


@dataclass
class OCRNetGenTrtEngineExpConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = STR_FIELD(value=None, default_value="")
    gpu_id: int = INT_FIELD(value=0)
    onnx_file: str = STR_FIELD(value=MISSING)
    trt_engine: Optional[str] = STR_FIELD(value=None)
    input_channel: int = INT_FIELD(value=3)
    input_width: int = INT_FIELD(value=100)
    input_height: int = INT_FIELD(value=32)
    opset_version: int = INT_FIELD(value=12)
    batch_size: int = INT_FIELD(value=-1)
    verbose: bool = BOOL_FIELD(value=False)
    tensorrt: TrtConfig = DATACLASS_FIELD(TrtConfig())


@dataclass
class OCRNetTrainExpConfig(TrainConfig):
    """Train experiment config."""

    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to pretrained model.",
        display="pretrained model path"
    )
    optim: OptimConfig = DATACLASS_FIELD(
        OptimConfig(),
        description="Configurable parameters for the optimizer.",
    )
    clip_grad_norm: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        description="The L2 magnitude of graident to be clipped in the training.",
        display="clip grad norm"
    )  # default = 5.0 for adadelta
    distributed_strategy: str = STR_FIELD(
        value="ddp",
        default_value="ddp",
        description="The distributed strategy for multi-gpu training.",
        display="distributed_strategy"
    )
    model_ema: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="The bool flag to enable model EMA.",
        display="model ema"
    )


@dataclass
class OCRNetInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to the TensorRT engine.",
        display="trt engine"
    )
    inference_dataset_dir: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="The absolute path to the inference dataset directory.",
        display="inference dataset dir"
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The inference batch size.",
        display="batch size"
    )
    input_width: int = INT_FIELD(
        value=100,
        default_value=100,
        description="Input width of the model.",
        display="input width"
    )
    input_height: int = INT_FIELD(
        value=32,
        default_value=32,
        description="Input height of the model.",
        display="input height"
    )


@dataclass
class OCRNetEvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    trt_engine: Optional[str] = STR_FIELD(value=None)
    test_dataset_dir: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="The absolute path to the test dataset directory.",
        display="test dataset dir"
    )
    test_dataset_gt_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to the test dataset ground truth file.",
        display="test dataset dir"
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The test batch size.",
        display="batch size"
    )
    input_width: int = INT_FIELD(
        value=100,
        default_value=100,
        description="Input width of the model.",
        display="input width"
    )
    input_height: int = INT_FIELD(
        value=32,
        default_value=32,
        description="Input height of the model.",
        display="input height"
    )


@dataclass
class OCRNetExportExpConfig:
    """Export experiment config."""

    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="The absolute path to the checkpoint.",
        display="checkpoint"
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to the results directory.",
        display="results dir"
    )
    onnx_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to the onnx file.",
        display="onnx file"
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="GPU ID.",
        display="gpu id"
    )


@dataclass
class OCRNetPruneExpConfig:
    """Prune experiment config."""

    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="The absolute path to the checkpoint.",
        display="checkpoint"
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to the results directory.",
        display="results dir"
    )
    pruned_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to the pruned model checkpoint.",
        display="pruned file"
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="GPU ID.",
        display="gpu id"
    )
    prune_setting: PruneConfig = DATACLASS_FIELD(
        PruneConfig(),
        description="Configurable parameters for the pruner.",
    )


@dataclass
class OCRNetConvertDatasetExpConfig:
    """Convert_dataset experiment config."""

    input_img_dir: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="The absolute path to the input images directory.",
        display="input img dir"
    )
    gt_file: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="The absolute path to the ground truth file.",
        display="gt file"
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to the results directory.",
        display="results dir"
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: OCRNetModelConfig = DATACLASS_FIELD(
        OCRNetModelConfig(),
        description="Configurable parameters for the model.",
    )
    dataset: OCRNetDatasetConfig = DATACLASS_FIELD(
        OCRNetDatasetConfig(),
        description="Configurable parameters for the dataset.",
    )
    train: OCRNetTrainExpConfig = DATACLASS_FIELD(
        OCRNetTrainExpConfig(),
        description="Configurable parameters for the training.",
    )
    evaluate: OCRNetEvalExpConfig = DATACLASS_FIELD(
        OCRNetEvalExpConfig(),
        description="Configurable parameters for the evaluation.",
    )
    export: OCRNetExportExpConfig = DATACLASS_FIELD(
        OCRNetExportExpConfig(),
        description="Configurable parameters for the export.",
    )
    inference: OCRNetInferenceExpConfig = DATACLASS_FIELD(
        OCRNetInferenceExpConfig(),
        description="Configurable parameters for the inference.",
    )
    prune: OCRNetPruneExpConfig = DATACLASS_FIELD(
        OCRNetPruneExpConfig(),
        description="Configurable parameters for the pruning.",
    )
    dataset_convert: OCRNetConvertDatasetExpConfig = DATACLASS_FIELD(
        OCRNetConvertDatasetExpConfig(),
        description="Configurable parameters for the dataset conversion.",
    )
    gen_trt_engine: OCRNetGenTrtEngineExpConfig = DATACLASS_FIELD(
        OCRNetGenTrtEngineExpConfig(),
        description="Configurable parameters for the TensorRT engine generation.",
    )
