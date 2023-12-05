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
from dataclasses import dataclass, field
from omegaconf import MISSING
from nvidia_tao_pytorch.pruning.prune_config import PruneConfig


@dataclass
class OCRNetModelConfig:
    """OCRNet model config."""

    TPS: bool = False  # Thin-Plate-Spline interpolation
    num_fiducial: int = 20  # number of keypoints for TPS
    backbone: str = "ResNet"  # [ResNet]
    feature_channel: int = 512
    sequence: str = "BiLSTM"  # [BiLSTM]
    hidden_size: int = 256
    prediction: str = "CTC"  # [Attn, CTC]
    quantize: bool = False
    input_width: int = 100
    input_height: int = 32
    input_channel: int = 1


@dataclass
class OptimConfig:
    """Optimizer config."""

    name: str = "adadelta"  # [adam, adadelta]
    lr: float = 1.0  # default value = 1.0 for adadelta
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_scheduler: str = "MultiStep"  # {AutoReduce, MultiStep}
    lr_monitor: str = "val_loss"  # {val_loss, train_loss}
    patience: int = 1
    min_lr: float = 1e-4
    lr_steps: List[int] = field(default_factory=lambda: [15, 25])
    lr_decay: float = 0.1


# TODO(tylerz): no augmentation from original implementation
@dataclass
class OCRNetAugmentationConfig:
    """Augmentation config."""

    keep_aspect_ratio: bool = False
    aug_prob: float = 0.0
    reverse_color_prob: float = 0.5
    rotate_prob: float = 0.5
    max_rotation_degree: int = 5
    blur_prob: float = 0.5
    gaussian_radius_list: Optional[List[int]] = field(default_factory=lambda: [1, 2, 3, 4])


@dataclass
class OCRNetDatasetConfig:
    """Dataset config."""

    train_dataset_dir: Optional[List[str]] = None
    train_gt_file: Optional[str] = None
    val_dataset_dir: Optional[str] = None
    val_gt_file: Optional[str] = None
    character_list_file: Optional[str] = None
    max_label_length: int = 25  # Shall we check it with output feature length ?
    batch_size: int = 32
    workers: int = 8
    augmentation: OCRNetAugmentationConfig = OCRNetAugmentationConfig()


@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_image_dir: List[str] = MISSING
    cal_cache_file: str = MISSING
    cal_batch_size: int = 1
    cal_batches: int = 1


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = "fp16"
    workspace_size: int = 1024
    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 1
    calibration: CalibrationConfig = CalibrationConfig()


@dataclass
class OCRNetGenTrtEngineExpConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = None
    gpu_id: int = 0
    onnx_file: str = MISSING
    trt_engine: Optional[str] = None
    input_channel: int = 3
    input_width: int = 100
    input_height: int = 32
    opset_version: int = 12
    batch_size: int = -1
    verbose: bool = False
    tensorrt: TrtConfig = TrtConfig()


@dataclass
class OCRNetTrainExpConfig:
    """Train experiment config."""

    results_dir: Optional[str] = None
    seed: int = 1111
    # TODO(tylerz): Update to use torch.distributed.launch for multi gpu training.
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_gpus: int = 1
    resume_training_checkpoint_path: Optional[str] = None
    pretrained_model_path: Optional[str] = None
    quantize_model_path: Optional[str] = None
    optim: OptimConfig = OptimConfig()
    num_epochs: int = 10
    clip_grad_norm: float = 5.0  # default = 5.0 for adadelta
    checkpoint_interval: int = 2
    validation_interval: int = 1
    distributed_strategy: str = "ddp"
    model_ema: bool = False


@dataclass
class OCRNetInferenceExpConfig:
    """Inference experiment config."""

    checkpoint: str = MISSING
    trt_engine: Optional[str] = None
    results_dir: Optional[str] = None
    gpu_id: int = 0
    inference_dataset_dir: str = MISSING
    batch_size: int = 1
    input_width: int = 100
    input_height: int = 32


@dataclass
class OCRNetEvalExpConfig:
    """Evaluation experiment config."""

    checkpoint: str = MISSING
    trt_engine: Optional[str] = None
    gpu_id: int = 0
    test_dataset_dir: str = MISSING
    test_dataset_gt_file: Optional[str] = None
    results_dir: Optional[str] = None
    batch_size: int = 1
    input_width: int = 100
    input_height: int = 32


@dataclass
class OCRNetExportExpConfig:
    """Export experiment config."""

    checkpoint: str = MISSING
    results_dir: Optional[str] = None
    onnx_file: Optional[str] = None
    gpu_id: int = 0


@dataclass
class OCRNetPruneExpConfig:
    """Prune experiment config."""

    checkpoint: str = MISSING
    results_dir: Optional[str] = None
    pruned_file: Optional[str] = None
    gpu_id: int = 0
    prune_setting: PruneConfig = PruneConfig()


@dataclass
class OCRNetConvertDatasetExpConfig:
    """Convert_dataset experiment config."""

    input_img_dir: str = MISSING
    gt_file: str = MISSING
    results_dir: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Experiment config."""

    model: OCRNetModelConfig = OCRNetModelConfig()
    dataset: OCRNetDatasetConfig = OCRNetDatasetConfig()
    train: OCRNetTrainExpConfig = OCRNetTrainExpConfig()
    evaluate: OCRNetEvalExpConfig = OCRNetEvalExpConfig()
    export: OCRNetExportExpConfig = OCRNetExportExpConfig()
    inference: OCRNetInferenceExpConfig = OCRNetInferenceExpConfig()
    prune: OCRNetPruneExpConfig = OCRNetPruneExpConfig()
    dataset_convert: OCRNetConvertDatasetExpConfig = OCRNetConvertDatasetExpConfig()
    gen_trt_engine: OCRNetGenTrtEngineExpConfig = OCRNetGenTrtEngineExpConfig()
    encryption_key: Optional[str] = None
    results_dir: str = MISSING
