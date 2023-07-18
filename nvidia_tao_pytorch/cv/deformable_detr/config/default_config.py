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
from omegaconf import MISSING


@dataclass
class DDDatasetConvertConfig:
    """Dataset Convert config."""

    input_source: Optional[str] = None
    data_root: Optional[str] = None
    results_dir: str = MISSING
    image_dir_name: Optional[str] = None
    label_dir_name: Optional[str] = None
    val_split: int = 0
    num_shards: int = 20
    num_partitions: int = 1
    partition_mode: Optional[str] = None
    image_extension: str = ".jpg"
    mapping_path: Optional[str] = None


@dataclass
class DDAugmentationConfig:
    """Augmentation config."""

    scales: List[int] = field(default_factory=lambda: [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                              metadata={"description": "Random Scales for Augmentation"})
    input_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406],
                                    metadata={"description": "Pixel mean value"})
    input_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225],
                                   metadata={"description": "Pixel Standard deviation value"})
    train_random_resize: List[int] = field(default_factory=lambda: [400, 500, 600],
                                           metadata={"description": "Training Random Resize"})
    horizontal_flip_prob: float = 0.5
    train_random_crop_min: int = 384
    train_random_crop_max: int = 600
    random_resize_max_size: int = 1333
    test_random_resize: int = 800
    fixed_padding: bool = True


@dataclass
class DDDatasetConfig:
    """Dataset config."""

    train_sampler: str = "default_sampler"
    train_data_sources: Optional[List[Dict[str, str]]] = None
    val_data_sources: Optional[List[Dict[str, str]]] = None
    test_data_sources: Optional[Dict[str, str]] = None
    infer_data_sources: Optional[Dict[str, str]] = None
    batch_size: int = 4
    workers: int = 8
    pin_memory: bool = True
    num_classes: int = 91
    dataset_type: str = "serialized"
    eval_class_ids: Optional[List[int]] = None
    augmentation: DDAugmentationConfig = DDAugmentationConfig()


@dataclass
class DDModelConfig:
    """Deformable DETR model config."""

    pretrained_backbone_path: Optional[str] = None
    backbone: str = "resnet_50"
    num_queries: int = 300
    num_feature_levels: int = 4
    return_interm_indices: List[int] = field(default_factory=lambda: [1, 2, 3, 4],
                                             metadata={"description": "Indices to return from backbone"})

    with_box_refine: bool = True
    cls_loss_coef: float = 2.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    focal_alpha: float = 0.25
    clip_max_norm: float = 0.1
    dropout_ratio: float = 0.3
    hidden_dim: int = 256
    nheads: int = 8
    enc_layers: int = 6
    dec_layers: int = 6
    dim_feedforward: int = 1024
    dec_n_points: int = 4
    enc_n_points: int = 4
    aux_loss: bool = True
    dilation: bool = False
    train_backbone: bool = True
    loss_types: List[str] = field(default_factory=lambda: ['labels', 'boxes'],
                                  metadata={"description": "Losses to be used during training"})
    backbone_names: List[str] = field(default_factory=lambda: ["backbone.0"],
                                      metadata={"description": "Backbone name"})
    linear_proj_names: List[str] = field(default_factory=lambda: ['reference_points', 'sampling_offsets'],
                                         metadata={"description": "Linear Projection names"})


@dataclass
class OptimConfig:
    """Optimizer config."""

    optimizer: str = "AdamW"
    monitor_name: str = "val_loss"  # {val_loss, train_loss}
    lr: float = 2e-4
    lr_backbone: float = 2e-5
    lr_linear_proj_mult: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_scheduler: str = "MultiStep"
    lr_steps: List[int] = field(default_factory=lambda: [40],
                                metadata={"description": "learning rate decay steps"})
    lr_step_size: int = 40
    lr_decay: float = 0.1


@dataclass
class DDTrainExpConfig:
    """Train experiment config."""

    num_gpus: int = 1
    num_nodes: int = 1
    resume_training_checkpoint_path: Optional[str] = None
    pretrained_model_path: Optional[str] = None
    validation_interval: int = 1
    clip_grad_norm: float = 0.1
    is_dry_run: bool = False
    results_dir: Optional[str] = None

    num_epochs: int = 50
    checkpoint_interval: int = 1
    optim: OptimConfig = OptimConfig()
    precision: str = "fp32"
    distributed_strategy: str = "ddp"
    activation_checkpoint: bool = True


@dataclass
class DDInferenceExpConfig:
    """Inference experiment config."""

    num_gpus: int = 1
    results_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    trt_engine: Optional[str] = None
    color_map: Dict[str, str] = MISSING
    conf_threshold: float = 0.5
    is_internal: bool = False
    input_width: Optional[int] = None
    input_height: Optional[int] = None


@dataclass
class DDEvalExpConfig:
    """Evaluation experiment config."""

    num_gpus: int = 1
    results_dir: Optional[str] = None
    input_width: Optional[int] = None
    input_height: Optional[int] = None
    checkpoint: Optional[str] = None
    trt_engine: Optional[str] = None
    conf_threshold: float = 0.0


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

    data_type: str = "FP32"
    workspace_size: int = 1024
    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 1
    calibration: CalibrationConfig = CalibrationConfig()


@dataclass
class DDExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = None
    gpu_id: int = 0
    checkpoint: str = MISSING
    onnx_file: str = MISSING
    on_cpu: bool = False
    input_channel: int = 3
    input_width: int = 960
    input_height: int = 544
    opset_version: int = 12
    batch_size: int = -1
    verbose: bool = False


@dataclass
class DDGenTrtEngineExpConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = None
    gpu_id: int = 0
    onnx_file: str = MISSING
    trt_engine: Optional[str] = None
    input_channel: int = 3
    input_width: int = 960
    input_height: int = 544
    opset_version: int = 12
    batch_size: int = -1
    verbose: bool = False
    tensorrt: TrtConfig = TrtConfig()


@dataclass
class ExperimentConfig:
    """Experiment config."""

    model: DDModelConfig = DDModelConfig()
    dataset: DDDatasetConfig = DDDatasetConfig()
    train: DDTrainExpConfig = DDTrainExpConfig()
    evaluate: DDEvalExpConfig = DDEvalExpConfig()
    inference: DDInferenceExpConfig = DDInferenceExpConfig()
    export: DDExportExpConfig = DDExportExpConfig()
    gen_trt_engine: DDGenTrtEngineExpConfig = DDGenTrtEngineExpConfig()
    encryption_key: Optional[str] = None
    results_dir: str = MISSING
