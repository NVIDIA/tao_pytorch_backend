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
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class ARModelConfig:
    """Action recognition model config."""

    model_type: str = "joint"
    backbone: str = "resnet_18"
    input_type: str = "2d"
    of_seq_length: int = 10
    of_pretrained_model_path: Optional[str] = None
    of_pretrained_num_classes: int = 0  # 0 means the pretrained model has the same classes number
    rgb_seq_length: int = 3
    rgb_pretrained_model_path: Optional[str] = None
    rgb_pretrained_num_classes: int = 0  # 0 means the pretrained model has the same classes number
    num_fc: int = 64
    joint_pretrained_model_path: Optional[str] = None
    sample_strategy: str = "random_interval"  # [random_interval, consecutive]
    sample_rate: int = 1
    imagenet_pretrained: bool = False  # Only for internal use. Will change to False when release
    # 0.0 for resnet18 2D on SHAD, 0.5 for I3D on HMDB51, 0.8 for ResNet3D on HMDB51
    dropout_ratio: float = 0.5
    input_width: int = 224
    input_height: int = 224


@dataclass
class OptimConfig:
    """Optimizer config."""

    lr: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_scheduler: str = "MultiStep"  # {AutoReduce, MultiStep}
    lr_monitor: str = "val_loss"  # {val_loss, train_loss}
    patience: int = 1
    min_lr: float = 1e-4
    lr_steps: List[int] = field(default_factory=lambda: [15, 25])
    lr_decay: float = 0.1


@dataclass
class ARAugmentationConfig:
    """Augmentation config."""

    train_crop_type: str = "random_crop"  # [random_crop, multi_scale_crop, no_crop]
    scales: List[float] = field(default_factory=lambda: [1])
    horizontal_flip_prob: float = 0.5
    rgb_input_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    rgb_input_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    of_input_mean: List[float] = field(default_factory=lambda: [0.5])
    of_input_std: List[float] = field(default_factory=lambda: [0.5])
    val_center_crop: bool = False
    crop_smaller_edge: int = 256


@dataclass
class ARDatasetConfig:
    """Dataset config."""

    train_dataset_dir: Optional[str] = None
    val_dataset_dir: Optional[str] = None
    label_map: Optional[Dict[str, int]] = None
    batch_size: int = 32
    workers: int = 8
    clips_per_video: int = 1
    augmentation_config: ARAugmentationConfig = ARAugmentationConfig()


@dataclass
class ARTrainExpConfig:
    """Train experiment config."""

    results_dir: Optional[str] = None
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    resume_training_checkpoint_path: Optional[str] = None
    optim: OptimConfig = OptimConfig()
    num_epochs: int = 10
    clip_grad_norm: float = 0.0
    checkpoint_interval: int = 5


@dataclass
class ARInferenceExpConfig:
    """Inference experiment config."""

    checkpoint: str = MISSING
    results_dir: Optional[str] = None
    gpu_id: int = 0
    inference_dataset_dir: str = MISSING
    batch_size: int = 1
    video_inf_mode: str = "center"  # [center, conv, all]
    video_num_segments: int = 1


@dataclass
class AREvalExpConfig:
    """Evaluation experiment config."""

    checkpoint: str = MISSING
    results_dir: Optional[str] = None
    gpu_id: int = 0
    test_dataset_dir: str = MISSING
    batch_size: int = 1
    video_eval_mode: str = "center"  # [center, conv, all]
    video_num_segments: int = 10


@dataclass
class ARExportExpConfig:
    """Export experiment config."""

    checkpoint: str = MISSING
    results_dir: Optional[str] = None
    onnx_file: Optional[str] = None
    gpu_id: int = 0
    batch_size: int = 1


@dataclass
class ExperimentConfig:
    """Experiment config."""

    model: ARModelConfig = ARModelConfig()
    dataset: ARDatasetConfig = ARDatasetConfig()
    train: ARTrainExpConfig = ARTrainExpConfig()
    evaluate: AREvalExpConfig = AREvalExpConfig()
    export: ARExportExpConfig = ARExportExpConfig()
    inference: ARInferenceExpConfig = ARInferenceExpConfig()
    encryption_key: Optional[str] = None
    results_dir: str = MISSING
