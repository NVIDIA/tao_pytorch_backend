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

"""Classification Default config file"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class ImgNormConfig:
    """Configuration parameters for Img Normalization."""

    mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])
    std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])
    to_rgb: bool = True


@dataclass
class TrainData:
    """Train Data Dataclass"""

    type: str = "ImageNet"
    data_prefix: Optional[str] = None
    pipeline: List[Any] = field(default_factory=lambda: [{"type": "RandomResizedCrop", "size": 224},
                                {"type": "RandomFlip", "flip_prob": 0.5, "direction": "horizontal"}])
    classes: Optional[str] = None


@dataclass
class ValData:
    """Validation Data Dataclass"""

    type: str = "ImageNet"
    data_prefix: Optional[str] = None
    ann_file: Optional[str] = None
    pipeline: List[Any] = field(default_factory=lambda: [{"type": "Resize", "size": (256, -1)},
                                {"type": "CenterCrop", "crop_size": 224}])
    classes: Optional[str] = None


@dataclass
class TestData:
    """Test Data Dataclass"""

    type: str = "ImageNet"
    data_prefix: Optional[str] = None
    ann_file: Optional[str] = None
    pipeline: List[Any] = field(default_factory=lambda: [{"type": "Resize", "size": (256, -1)},
                                                         {"type": "CenterCrop", "crop_size": 224}])
    classes: Optional[str] = None


@dataclass
class DataConfig:
    """Data Config"""

    samples_per_gpu: int = 1
    workers_per_gpu: int = 2
    train: TrainData = TrainData()
    val: ValData = ValData()
    test: TestData = TestData()


@dataclass
class DatasetConfig:
    """Dataset config."""

    img_norm_cfg: ImgNormConfig = ImgNormConfig()
    data: DataConfig = DataConfig()
    sampler: Optional[Dict[Any, Any]] = None  # Allowed sampler : RepeatAugSampler


@dataclass
class DistParams:
    """Distribution Parameters"""

    backend: str = "nccl"


@dataclass
class RunnerConfig:
    """Configuration parameters for Runner."""

    type: str = "TAOEpochBasedRunner"  # Currently We support only Epochbased Runner - Non configurable
    max_epochs: int = 20  # Set this if Epoch based runner


@dataclass
class CheckpointConfig:
    """Configuration parameters for Checkpointing."""

    interval: int = 1  # Epochs or Iterations accordingly
    by_epoch: bool = True  # By default it trains by iters


# Default Runtime Config
@dataclass
class LogConfig:
    """Configuration parameters for Logging."""

    interval: int = 1000
    log_dir: str = "logs"  # Make sure this directory is created


# Optim and Schedule Config
@dataclass
class ValidationConfig:
    """Validation Config."""

    interval: int = 100


@dataclass
class ParamwiseConfig:
    """Configuration parameters for Parameters."""

    pos_block: Dict[str, float] = field(default_factory=lambda: {"decay_mult": 0.0})
    norm: Dict[str, float] = field(default_factory=lambda: {"decay_mult": 0.0})
    head: Dict[str, float] = field(default_factory=lambda: {"lr_mult": 10.0})


@dataclass
class EvaluationConfig:
    """Evaluation Config."""

    interval: int = 1
    metric: str = "accuracy"


@dataclass
class TrainConfig:
    """Train Config."""

    checkpoint_config: CheckpointConfig = CheckpointConfig()
    optimizer: Dict[Any, Any] = field(default_factory=lambda: {"type": 'AdamW',
                                                               "lr": 10e-4,
                                                               "weight_decay": 0.05})
    paramwise_cfg: Optional[Dict[Any, Any]] = None  # Not a must - needs to be provided in yaml
    optimizer_config: Dict[Any, Any] = field(default_factory=lambda: {'grad_clip': None})  # Gradient Accumulation and grad clip
    lr_config: Dict[Any, Any] = field(default_factory=lambda: {"policy": 'CosineAnnealing',
                                                               "min_lr": 10e-4, "warmup": "linear",
                                                               "warmup_iters": 5,
                                                               "warmup_ratio": 0.01,
                                                               "warmup_by_epoch": True})
    runner: RunnerConfig = RunnerConfig()
    logging: LogConfig = LogConfig()  # By default we add logging
    evaluation: EvaluationConfig = EvaluationConfig()  # Does not change
    find_unused_parameters: bool = False  # Does not change
    resume_training_checkpoint_path: Optional[str] = None
    validate: bool = False
    # This param can be omitted if init_cfg is used in model_cfg. Both does same thing.
    load_from: Optional[str] = None  # If they want to load the weights till head
    custom_hooks: List[Any] = field(default_factory=lambda: [])


# Experiment Common Configs
@dataclass
class ExpConfig:
    """Overall Exp Config for Cls."""

    manual_seed: int = 47
    # If needed, the next line can be commented
    MASTER_ADDR: str = "127.0.0.1"
    MASTER_PORT: int = 631


@dataclass
class TrainExpConfig:
    """Train experiment config."""

    exp_config: ExpConfig = ExpConfig()
    validate: bool = False
    train_config: TrainConfig = TrainConfig()  # Could change across networks
    num_gpus: int = 1  # non configurable here
    results_dir: Optional[str] = None


@dataclass
class InferenceExpConfig:
    """Inference experiment config."""

    num_gpus: int = 1  # non configurable here
    batch_size: int = 1
    checkpoint: Optional[str] = None
    trt_engine: Optional[str] = None
    exp_config: ExpConfig = ExpConfig()
    results_dir: Optional[str] = None


@dataclass
class EvalExpConfig:
    """Inference experiment config."""

    num_gpus: int = 1  # non configurable here
    batch_size: int = 1
    checkpoint: Optional[str] = None
    trt_engine: Optional[str] = None
    exp_config: ExpConfig = ExpConfig()
    topk: int = 1  # Configurable
    results_dir: Optional[str] = None


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = "FP32"
    workspace_size: int = 1024
    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 1


@dataclass
class ExportExpConfig:
    """Export experiment config."""

    verify: bool = False
    opset_version: int = 12
    checkpoint: Optional[str] = None
    input_channel: int = 3
    input_width: int = 224
    input_height: int = 224
    onnx_file: Optional[str] = None
    results_dir: Optional[str] = None


@dataclass
class HeadConfig:
    """Head Config"""

    type: str = 'LinearClsHead'
    num_classes: int = 1000
    in_channels: int = 448  # Mapped to differenct channels based according to the backbone used in the fan_model.py
    custom_args: Optional[Dict[Any, Any]] = None
    loss: Dict[Any, Any] = field(default_factory=lambda: {"type": 'CrossEntropyLoss'})
    topk: List[int] = field(default_factory=lambda: [1, ])


@dataclass
class InitCfg:
    """Init Config"""

    type: str = "Pretrained"
    checkpoint: Optional[str] = None
    prefix: Optional[str] = None  # E.g., backbone


@dataclass
class BackboneConfig:
    """Configuration parameters for Backbone."""

    type: str = "fan_tiny_8_p4_hybrid"
    custom_args: Optional[Dict[Any, Any]] = None
    freeze: bool = False
    pretrained: Optional[str] = None


@dataclass
class TrainAugCfg:
    """Arguments for Train Config"""

    augments: Optional[List[Dict[Any, Any]]] = None


@dataclass
class ModelConfig:
    """Cls model config."""

    type: str = "ImageClassifier"
    backbone: BackboneConfig = BackboneConfig()
    neck: Optional[Dict[Any, Any]] = None
    head: HeadConfig = HeadConfig()
    init_cfg: InitCfg = InitCfg()  # No change
    train_cfg: TrainAugCfg = TrainAugCfg()


@dataclass
class GenTrtEngineExpConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = None
    gpu_id: int = 0
    onnx_file: Optional[str] = None
    trt_engine: Optional[str] = None
    input_channel: int = 3
    input_width: int = 224
    input_height: int = 224
    opset_version: int = 12
    batch_size: int = -1
    verbose: bool = False
    tensorrt: TrtConfig = TrtConfig()


@dataclass
class ExperimentConfig:
    """Experiment config."""

    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    train: TrainExpConfig = TrainExpConfig()
    evaluate: EvalExpConfig = EvalExpConfig()
    inference: InferenceExpConfig = InferenceExpConfig()
    gen_trt_engine: GenTrtEngineExpConfig = GenTrtEngineExpConfig()
    export: ExportExpConfig = ExportExpConfig()
    results_dir: str = MISSING
