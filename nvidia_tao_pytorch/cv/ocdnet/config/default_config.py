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

from typing import Optional, List
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class ModelConfig:
    """Model config."""

    backbone: str = "deformable_resnet18"
    pretrained: bool = False
    in_channels: int = 3
    neck: str = "FPN"
    inner_channels: int = 256
    head: str = "DBHead"
    out_channels: int = 2
    k: int = 50
    load_pruned_graph: bool = MISSING
    pruned_graph_path: Optional[str] = None
    pretrained_model_path: Optional[str] = None
    enlarge_feature_map_size: bool = False
    activation_checkpoint: bool = False
    quant: bool = False
    fuse_qkv_proj: bool = True


@dataclass
class Optimargs:
    """Optimargs config."""

    lr: float = 0.001
    weight_decay: float = 0.0
    amsgrad: bool = True
    momentum: float = 0.0
    eps: float = 1e-8


@dataclass
class Optimizer:
    """Optimizer config."""

    type: str = "Adam"
    args: Optimargs = Optimargs()


@dataclass
class Loss:
    """Loss config."""

    type: str = "DBLoss"
    alpha: int = 5
    beta: int = 10
    ohem_ratio: int = 3
    eps: float = 1e-6


@dataclass
class Postprocessingargs:
    """Postprocessingargs config."""

    thresh: float = MISSING
    box_thresh: float = MISSING
    max_candidates: int = MISSING
    unclip_ratio: float = MISSING


@dataclass
class Postprocessing:
    """Postprocessing config."""

    type: str = "SegDetectorRepresenter"
    args: Postprocessingargs = Postprocessingargs()


@dataclass
class Metricargs:
    """Metricargs config."""

    is_output_polygon: bool = MISSING


@dataclass
class Metric:
    """Metric config."""

    type: str = "QuadMetric"
    args: Metricargs = Metricargs()


@dataclass
class LRSchedulerargs:
    """LRSchedulerargs config."""

    warmup_epoch: int = MISSING


@dataclass
class LRScheduler:
    """LRScheduler config."""

    type: str = "WarmupPolyLR"
    args: LRSchedulerargs = LRSchedulerargs()


@dataclass
class Trainer:
    """Trainer config."""

    is_output_polygon: bool = False
    warmup_epoch: int = 3
    seed: int = 2
    log_iter: int = 10
    clip_grad_norm: float = 5.0
    show_images_iter: int = 50
    tensorboard: bool = False


@dataclass
class Trainargs:
    """Train args config."""

    img_mode: str = "BGR"
    filter_keys: List[str] = field(default_factory=lambda: ['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags', 'shape'])
    ignore_tags: List[str] = field(default_factory=lambda: ['*', '###'])
    pre_processes: Optional[List[str]] = None


@dataclass
class Dataloader:
    """Train args config."""

    batch_size: int = 32
    shuffle: bool = True
    pin_memory: bool = False
    num_workers: int = 0
    collate_fn: Optional[str] = ""


@dataclass
class TrainDataset:
    """Train Dataset config."""

    data_name: str = "ICDAR2015Dataset"
    data_path: List[str] = MISSING
    args: Trainargs = Trainargs()
    loader: Dataloader = Dataloader()


@dataclass
class Validateargs:
    """Validate args config."""

    img_mode: str = "BGR"
    filter_keys: List[str] = field(default_factory=lambda: [''])
    ignore_tags: List[str] = field(default_factory=lambda: ['*', '###'])
    pre_processes: Optional[List[str]] = None


@dataclass
class Validateloader:
    """Validate args config."""

    batch_size: int = 1
    shuffle: bool = False
    pin_memory: bool = False
    num_workers: int = 0
    collate_fn: Optional[str] = "ICDARCollateFN"


@dataclass
class ValidateDataset:
    """Validate Dataset config."""

    data_name: str = "ICDAR2015Dataset"
    data_path: List[str] = MISSING
    args: Validateargs = Validateargs()
    loader: Validateloader = Validateloader()


@dataclass
class DataConfig:
    """Dataset config."""

    train_dataset: TrainDataset = TrainDataset()
    validate_dataset: ValidateDataset = ValidateDataset()


@dataclass
class TrainConfig:
    """Train experiment config."""

    results_dir: Optional[str] = None
    resume_training_checkpoint_path: Optional[str] = None
    num_epochs: int = 50
    checkpoint_interval: int = 1
    validation_interval: int = 1
    gpu_id: List[int] = field(default_factory=lambda: [0])
    post_processing: Postprocessing = Postprocessing()
    metric: Metric = Metric()
    trainer: Trainer = Trainer()
    loss: Loss = Loss()
    optimizer: Optimizer = Optimizer()
    lr_scheduler: LRScheduler = LRScheduler()
    precision: str = "fp32"
    distributed_strategy: str = "ddp"
    is_dry_run: bool = False
    model_ema: bool = False
    model_ema_decay: float = 0.9999


@dataclass
class InferenceConfig:
    """Inference experiment config."""

    results_dir: Optional[str] = None
    checkpoint: str = MISSING
    trt_engine: Optional[str] = None
    input_folder: str = MISSING
    width: int = MISSING
    height: int = MISSING
    img_mode: str = MISSING
    polygon: bool = True
    show: bool = False
    gpu_id: int = 0
    post_processing: Postprocessing = Postprocessing()


@dataclass
class EvalConfig:
    """Evaluation experiment config."""

    results_dir: Optional[str] = None
    checkpoint: str = MISSING
    trt_engine: Optional[str] = None
    gpu_id: int = 0
    batch_size: int = 1
    post_processing: Postprocessing = Postprocessing()
    metric: Metric = Metric()


@dataclass
class PruneConfig:
    """Prune experiment config."""

    results_dir: Optional[str] = None
    checkpoint: str = MISSING
    gpu_id: int = 0
    ch_sparsity: float = 0.1
    round_to: int = 32
    p: int = 2
    verbose: bool = False


@dataclass
class ExportConfig:
    """Export experiment config."""

    results_dir: Optional[str] = None
    checkpoint: str = MISSING
    onnx_file: Optional[str] = None
    gpu_id: int = 0
    width: int = MISSING
    height: int = MISSING
    opset_version: int = 11
    verbose: bool = False


@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_image_dir: str = MISSING
    cal_cache_file: str = MISSING
    cal_batch_size: int = 1
    cal_num_batches: int = 1


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = "FP32"
    workspace_size: int = 1024
    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 1
    calibration: CalibrationConfig = CalibrationConfig()
    layers_precision: Optional[List[str]] = None


@dataclass
class GenTrtEngineExpConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = None
    gpu_id: int = 0
    onnx_file: str = MISSING
    trt_engine: str = MISSING
    width: int = MISSING
    height: int = MISSING
    img_mode: str = "BGR"
    tensorrt: TrtConfig = TrtConfig()


@dataclass
class ExperimentConfig:
    """Experiment config."""

    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()
    evaluate: EvalConfig = EvalConfig()
    dataset: DataConfig = DataConfig()
    export: ExportConfig = ExportConfig()
    gen_trt_engine: GenTrtEngineExpConfig = GenTrtEngineExpConfig()
    inference: InferenceConfig = InferenceConfig()
    prune: PruneConfig = PruneConfig()
    name: str = MISSING
    num_gpus: int = 1
    results_dir: str = MISSING
