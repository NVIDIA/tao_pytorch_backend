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
from dataclasses import dataclass, field


@dataclass
class PCModelConfig:
    """Pose classification model config."""

    model_type: str = "ST-GCN"
    pretrained_model_path: Optional[str] = None
    input_channels: int = 3
    dropout: float = 0.5
    graph_layout: str = "nvidia"  # [nvidia, openpose, human3.6m, ntu-rgb+d, ntu_edge, coco]
    graph_strategy: str = "spatial"  # [uniform, distance, spatial]
    edge_importance_weighting: bool = True


@dataclass
class OptimConfig:
    """Optimizer config."""

    optimizer_type: str = "torch.optim.SGD"
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 0.0001
    lr_scheduler: str = "MultiStep"  # {AutoReduce, MultiStep}
    lr_monitor: str = "val_loss"  # {val_loss, train_loss}
    patience: int = 1
    min_lr: float = 1e-4
    lr_steps: List[int] = field(default_factory=lambda: [10, 60])
    lr_decay: float = 0.1


@dataclass
class SkeletonDatasetConfig:
    """Skeleton dataset config."""

    data_path: Optional[str] = None
    label_path: Optional[str] = None


@dataclass
class PCDatasetConfig:
    """Dataset config."""

    train_dataset: SkeletonDatasetConfig = SkeletonDatasetConfig()
    val_dataset: SkeletonDatasetConfig = SkeletonDatasetConfig()
    num_classes: int = 6
    label_map: Optional[Dict[str, int]] = None
    random_choose: bool = False
    random_move: bool = False
    window_size: int = -1
    batch_size: int = 64
    num_workers: int = 1


@dataclass
class PCTrainExpConfig:
    """Train experiment config."""

    results_dir: Optional[str] = None
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_gpus: int = 1
    resume_training_checkpoint_path: Optional[str] = None
    optim: OptimConfig = OptimConfig()
    num_epochs: int = 70
    checkpoint_interval: int = 5
    grad_clip: float = 0.0


@dataclass
class PCInferenceExpConfig:
    """Inference experiment config."""

    results_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    output_file: Optional[str] = None
    test_dataset: SkeletonDatasetConfig = SkeletonDatasetConfig()
    gpu_id: int = 0


@dataclass
class PCEvalExpConfig:
    """Evaluation experiment config."""

    results_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    test_dataset: SkeletonDatasetConfig = SkeletonDatasetConfig()
    gpu_id: int = 0


@dataclass
class PCExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    onnx_file: Optional[str] = None
    gpu_id: int = 0


@dataclass
class PCDatasetConvertExpConfig:
    """Dataset conversion experiment config."""

    results_dir: Optional[str] = None
    data: Optional[str] = None
    pose_type: str = "3dbp"  # [3dbp, 25dbp, 2dbp]
    num_joints: int = 34
    input_width: int = 1920
    input_height: int = 1080
    focal_length: float = 1200.0
    sequence_length_max: int = 300
    sequence_length_min: int = 10
    sequence_length: int = 100
    sequence_overlap: float = 0.5


@dataclass
class ExperimentConfig:
    """Experiment config."""

    results_dir: Optional[str] = None
    encryption_key: Optional[str] = None
    model: PCModelConfig = PCModelConfig()
    dataset: PCDatasetConfig = PCDatasetConfig()
    train: PCTrainExpConfig = PCTrainExpConfig()
    inference: PCInferenceExpConfig = PCInferenceExpConfig()
    evaluate: PCEvalExpConfig = PCEvalExpConfig()
    export: PCExportExpConfig = PCExportExpConfig()
    dataset_convert: PCDatasetConvertExpConfig = PCDatasetConvertExpConfig()
