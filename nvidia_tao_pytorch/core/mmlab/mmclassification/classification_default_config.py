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
from dataclasses import dataclass

from nvidia_tao_pytorch.core.mmlab.mmclassification.model_params_mapping import map_params
from nvidia_tao_pytorch.config.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)

SUPPORTED_BACKBONES = [
    *list(map_params['head']['in_channels'].keys())
]


@dataclass
class ImgNormConfig:
    """Configuration parameters for Img Normalization."""

    mean: List[float] = LIST_FIELD([123.675, 116.28, 103.53], default_value=[123.675, 116.28, 103.53], description="Mean for the augmentation", display_name="Mean")
    std: List[float] = LIST_FIELD([58.395, 57.12, 57.375], default_value=[58.395, 57.12, 57.375], description="Standard deviation for the augmentation", display_name="Standard Deviation")
    to_rgb: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to convert to rgb")


@dataclass
class TrainData:
    """Train Data Dataclass"""

    type: str = STR_FIELD(value="ImageNet", default_value="", description="Type of training data")
    data_prefix: Optional[str] = STR_FIELD(value="", default_value="", description="Dataset directory path")
    pipeline: List[Any] = LIST_FIELD([{"type": "RandomResizedCrop", "scale": 224}, {"type": "RandomFlip", "prob": 0.5, "direction": "horizontal"},], description="Augmentation pipeline")
    classes: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to text file containing class names")


@dataclass
class ValData:
    """Validation Data Dataclass"""

    type: str = STR_FIELD(value="ImageNet", default_value="", description="Type of validation data")
    data_prefix: Optional[str] = STR_FIELD(value=None, default_value="", description="Dataset directory path")
    ann_file: Optional[str] = STR_FIELD(value=None, default_value="", description="Data annotation file path")
    pipeline: List[Any] = LIST_FIELD([{"type": "Resize", "scale": 224}, {"type": "CenterCrop", "crop_size": 224}], description="Augmentation pipeline")
    classes: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to text file containing class names")


@dataclass
class TestData:
    """Test Data Dataclass"""

    type: str = STR_FIELD(value="ImageNet", default_value="", description="Type of test data")
    data_prefix: Optional[str] = STR_FIELD(value=None, default_value="", description="Dataset directory path")
    ann_file: Optional[str] = STR_FIELD(value=None, default_value="", description="Data annotation file path")
    pipeline: List[Any] = LIST_FIELD([{"type": "Resize", "scale": 224}, {"type": "CenterCrop", "crop_size": 224}], description="Augmentation pipeline")
    classes: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to text file containing class names")


@dataclass
class DataConfig:
    """Data Config"""

    samples_per_gpu: int = INT_FIELD(value=1, valid_min=1, valid_max="inf", description="samples per gpu", display_name="samples per gpu", automl_enabled="TRUE")
    workers_per_gpu: int = INT_FIELD(value=2, valid_min=0, valid_max="inf", description="Workers", display_name="Workers", automl_enabled="TRUE")
    train: TrainData = DATACLASS_FIELD(TrainData())
    val: ValData = DATACLASS_FIELD(ValData())
    test: TestData = DATACLASS_FIELD(TestData())


@dataclass
class DatasetConfig:
    """Dataset config."""

    img_norm_cfg: ImgNormConfig = DATACLASS_FIELD(ImgNormConfig())
    data: DataConfig = DATACLASS_FIELD(DataConfig())
    sampler: Dict[Any, Any] = DICT_FIELD({"type": "DefaultSampler", "shuffle": True}, description="Dataset Sampler")  # Allowed sampler : RepeatAugSampler
    pin_memory: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to pin memory")  # Newly supported
    persistent_workers: bool = BOOL_FIELD(value=True, default_value=True, description="Flag for persistent workers")  # Newly supported
    collate_fn: Dict[Any, Any] = DICT_FIELD({"type": "default_collate"}, description="collate function")  # Does not change


@dataclass
class DistParams:
    """Distribution Parameters"""

    backend: str = STR_FIELD(value="nccl", default_value="nccl", description="type of backend")


@dataclass
class RunnerConfig:
    """Configuration parameters for Runner."""

    type: str = STR_FIELD(value="TAOEpochBasedRunner", default_value="TAOEpochBasedRunner", description="Runner config")  # Currently We support only Epochbased Runner - Non configurable
    max_epochs: int = INT_FIELD(value=20, default_value=40, valid_min=1, valid_max="inf", parent_param="TRUE", automl_enabled="TRUE", description="Max epochs")  # Set this if Epoch based runner
    auto_scale_lr_bs: int = INT_FIELD(value=1024, description="auto scale lr batch size")


@dataclass
class CheckpointConfig:
    """Configuration parameters for Checkpointing."""

    interval: int = INT_FIELD(value=1, default_value=1, valid_min=1, valid_max="inf", math_cond="<" + str(RunnerConfig.max_epochs), description="checkpointing interval")  # Epochs or Iterations accordingly
    by_epoch: bool = BOOL_FIELD(value=True, description="Flag to enable by epoch")  # By default it trains by iters


# Default Runtime Config
@dataclass
class LogConfig:
    """Configuration parameters for Logging."""

    interval: int = INT_FIELD(value=1000, valid_min=1, description="logging interval")
    log_dir: str = STR_FIELD(value="logs", description="logging directory")  # Make sure this directory is created


# Optim and Schedule Config
@dataclass
class ValidationConfig:
    """Validation Config."""

    interval: int = INT_FIELD(value=100, valid_min=1, description="validation interval")


@dataclass
class ParamwiseConfig:
    """Configuration parameters for Parameters."""

    pos_block: Dict[str, float] = DICT_FIELD(hashMap={"decay_mult": 0.0}, description="pos_block")
    norm: Dict[str, float] = DICT_FIELD(hashMap={"decay_mult": 0.0}, description="norm")
    head: Dict[str, float] = DICT_FIELD(hashMap={"lr_mult": 10.0}, description="head")


@dataclass
class EvaluationConfig:
    """Evaluation Config."""

    interval: int = INT_FIELD(value=1, valid_min=1, description="Evaluation interval")
    metric: str = STR_FIELD(value="accuracy", description="Evaluation metric")


@dataclass
class EnvConfig:
    """Environment Config."""

    cudnn_benchmark: bool = BOOL_FIELD(value=False, description="cudnn_benchmark")
    mp_cfg: Dict[Any, Any] = DICT_FIELD(hashMap={"mp_start_method": "fork", "opencv_num_threads": 0}, description="mp configuration")
    dist_cfg: Dict[Any, Any] = DICT_FIELD(hashMap={"backend": "nccl"}, description="distributed configuration")


@dataclass
class TrainConfig:
    """Train Config."""

    checkpoint_config: CheckpointConfig = DATACLASS_FIELD(CheckpointConfig())
    optimizer: Dict[Any, Any] = DATACLASS_FIELD({"type": "AdamW", "lr": 10e-4, "weight_decay": 0.05}, description="Optimizer config")
    paramwise_cfg: Optional[Dict[Any, Any]] = DICT_FIELD(None, default_value=None, description="Parameter configurations")  # Not a must - needs to be provided in yaml
    optimizer_config: Dict[Any, Any] = DICT_FIELD(hashMap={"grad_clip": None}, default_value={"grad_clip": None}, description="optimizer configuration")  # Gradient Accumulation and grad clip
    lr_config: Dict[Any, Any] = DICT_FIELD(hashMap={"type": "CosineAnnealingLR"}, default_value={"type": "CosineAnnealingLR", "T_max": 95, "by_epoch": True, "begin": 5, "end": 100}, description="learning rate config")
    runner: RunnerConfig = DATACLASS_FIELD(RunnerConfig())
    logging: LogConfig = DATACLASS_FIELD(LogConfig())  # By default we add logging
    evaluation: EvaluationConfig = DATACLASS_FIELD(EvaluationConfig())  # Does not change
    find_unused_parameters: bool = BOOL_FIELD(value=False, default_value=False, description="flag to find unused parameters")  # Does not change
    resume_training_checkpoint_path: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to checkpoint to resume training")
    validate: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to validate during training")  # Does not change

    # This param can be omitted if init_cfg is used in model_cfg. Both does same thing.
    load_from: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to checkpoint")  # If they want to load the weights till head
    custom_hooks: List[Any] = LIST_FIELD([], default_value=[{"type": "EMAHook", "momentum": "4e-5", "priority": "ABOVE_NORMAL"}], description="custom hooks for training")
    default_hooks: Dict[Any, Any] = DICT_FIELD(hashMap={"timer": {"type": "IterTimerHook"}, "param_scheduler": {"type": "ParamSchedulerHook"}, "checkpoint": {"type": "CheckpointHook", "interval": 1}, "logger": {"type": "LoggerHook"}, "visualization": {"type": "VisualizationHook", "enable": False}, "sampler_seed": {"type": "DistSamplerSeedHook"}}, description="default hooks")  # Does not change for old spec
    resume: bool = BOOL_FIELD(value=False, default_value=False, description="Flag to resume training")  # Not exposed


# Experiment Common Configs
@dataclass
class ExpConfig:
    """Overall Exp Config for Cls."""

    manual_seed: int = INT_FIELD(value=47, default_value=47, description="manual seed")
    # If needed, the next line can be commented
    MASTER_ADDR: str = STR_FIELD(value="127.0.0.1", description="Master node address")
    MASTER_PORT: int = INT_FIELD(value=631, description="master port number")
    env_config: EnvConfig = DATACLASS_FIELD(EnvConfig())
    deterministic: bool = BOOL_FIELD(value=False, default_value=False, description="Flag for deterministic")


@dataclass
class TrainExpConfig:
    """Train experiment config."""

    exp_config: ExpConfig = DATACLASS_FIELD(ExpConfig())
    validate: bool = BOOL_FIELD(value=False, default_value=False, description="Flag to validate")
    train_config: TrainConfig = DATACLASS_FIELD(TrainConfig())  # Could change across networks
    num_gpus: int = INT_FIELD(value=1, valid_min=1, valid_max="inf", description="Number of GPUs")  # non configurable here
    gpu_ids: List[int] = LIST_FIELD([0], description="GPU ID")
    results_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Results directory", display_name="Results directory")


@dataclass
class InferenceExpConfig:
    """Inference experiment config."""

    exp_config: ExpConfig = DATACLASS_FIELD(ExpConfig())
    num_gpus: int = INT_FIELD(value=1, valid_min=1, valid_max="inf", description="Number of GPUs")  # non configurable here
    gpu_ids: List[int] = LIST_FIELD([0], description="GPU ID")
    batch_size: int = INT_FIELD(value=1, default_value=1, valid_min=1, valid_max="inf", description="Batch size", display_name="Batch Size", automl_enabled="TRUE")
    checkpoint: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to checkpoint file", display_name="Path to checkpoint file")
    trt_engine: Optional[str] = STR_FIELD(value=None, description="TRT engine", display_name="TRT engine")
    exp_config: ExpConfig = DATACLASS_FIELD(ExpConfig())
    results_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Results directory", display_name="Results directory")


@dataclass
class EvalExpConfig:
    """Inference experiment config."""

    exp_config: ExpConfig = DATACLASS_FIELD(ExpConfig())
    num_gpus: int = INT_FIELD(value=1, valid_min=1, valid_max="inf", description="Number of GPUs")  # non configurable here
    gpu_ids: List[int] = LIST_FIELD([0], description="GPU ID")
    batch_size: int = INT_FIELD(value=1, default_value=8, valid_min=1, valid_max="inf", description="Batch size", display_name="Batch Size", automl_enabled="TRUE")
    checkpoint: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to checkpoint file", display_name="Path to checkpoint file")
    trt_engine: Optional[str] = STR_FIELD(value=None, description="TRT engine", display_name="TRT engine")
    exp_config: ExpConfig = DATACLASS_FIELD(ExpConfig())
    topk: int = INT_FIELD(value=1, valid_min=1, valid_max="inf", description="topk accuracy")  # Configurable
    results_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Results directory", display_name="Results directory")


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = STR_FIELD(value="fp16", valid_options="fp32,fp16,int8", popular="yes", display_name="min_batch_size", description="TensorRT data type")
    workspace_size: int = INT_FIELD(value=1024, display_name="max_batch_size", description="maximum workspace size of TensorRT engine (default 1<<30). If meet with out-of-memory issue, please increase the workspace size accordingly.")
    min_batch_size: int = INT_FIELD(value=1, display_name="opt_batch_size", popular="yes", description="maximum TensorRT engine batch size (default 1). If meet with out-of-memory issue, please decrease the batch size accordingly.",)
    opt_batch_size: int = INT_FIELD(value=1, display_name="data_type", popular="yes", description="maximum TensorRT engine batch size (default 1). If meet with out-of-memory issue, please decrease the batch size accordingly.",)
    max_batch_size: int = INT_FIELD(value=1, display_name="max_workspace_size", popular="yes", description="maximum TensorRT engine batch size (default 1). If meet with out-of-memory issue, please decrease the batch size accordingly.",)


@dataclass
class ExportExpConfig:
    """Export experiment config."""

    verify: bool = BOOL_FIELD(value=False, description="Flag to verify export")
    opset_version: int = INT_FIELD(value=12, default_value=12, valid_min=11, valid_max=17, display_name="opset version", description="""Operator set version of the ONNX model used to generate the TensorRT engine.""")
    checkpoint: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to checkpoint file", display_name="Path to checkpoint file")
    input_channel: int = INT_FIELD(value=3, default_value=3, description="Input channel", display_name="Input channel")
    input_width: int = INT_FIELD(value=224, default_value=224, description="Input width", display_name="Input width")
    input_height: int = INT_FIELD(value=224, default_value=224, description="Input height", display_name="Input height")
    onnx_file: Optional[str] = STR_FIELD(value=None, default_value="", description="ONNX file", display_name="ONNX file")
    results_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Results directory", display_name="Results directory")


@dataclass
class LRHeadConfig:
    """Logistic Regression Head Config"""

    C: float = FLOAT_FIELD(0.316, default_value=0.316, valid_min=0, valid_max="inf", automl_enabled="TRUE", description="C parameter for Logistic Regression")
    max_iter: int = INT_FIELD(value=5000, default_value=10000, valid_min=0, valid_max="inf", automl_enabled="TRUE", description="max iterations for LR head")
    class_weight: Optional[str] = STR_FIELD(value=None, default_value="balanced", description="Class weights for LR head")
    solver: Optional[str] = STR_FIELD(value="lbfgs", valid_options="lbfgs,liblinear,newton-cg,newton-cholesky,sag,saga", automl_enabled="TRUE", description="solver")
    hpo: bool = BOOL_FIELD(value=False, description="Flag to enable hyperparameter search during training")
    cs_tune: List[float] = LIST_FIELD(arrList=[0.001, 0.01,  0.316, 1, 10, 1000, 10000], description="List of C values to search from durin training")
    criteria: str = STR_FIELD(value="accuracy", description="Criteria for HPO")


@dataclass
class HeadConfig:
    """Head Config"""

    type: str = STR_FIELD(value="TAOLinearClsHead", value_type="ordered", valid_options="TAOLinearClsHead,LogisticRegressionHead", description="Type of classification head")
    binary: bool = BOOL_FIELD(value=False, description="Flag to specify binary classification")
    num_classes: int = INT_FIELD(value=1000, default_value=20, valid_min=2, valid_max="inf", description="Number of classes")
    in_channels: int = INT_FIELD(value=448, description="Number of backbone input channels to head")  # Mapped to differenct channels based according to the backbone used in the fan_model.py
    custom_args: Optional[Dict[Any, Any]] = DICT_FIELD(None, default_value=None, description="custom head arguments")
    loss: Dict[Any, Any] = DICT_FIELD(hashMap={"type": "CrossEntropyLoss"}, default_value={"type": "CrossEntropyLoss", "loss_weight": 1, "use_soft": False}, description="Loss configs")
    topk: List[int] = LIST_FIELD([1], description="k value for Topk accuracy")
    lr_head: LRHeadConfig = DATACLASS_FIELD(LRHeadConfig())


@dataclass
class InitCfg:
    """Init Config"""

    type: str = STR_FIELD(value="Pretrained", description="Initialisation config")
    checkpoint: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to checkpoint file", display_name="Path to checkpoint file")
    prefix: Optional[str] = STR_FIELD(value=None, default_value="", description="prefix for loading backbone weights")  # E.g., backbone


@dataclass
class BackboneConfig:
    """Configuration parameters for Backbone."""

    type: str = STR_FIELD(value="fan_tiny_8_p4_hybrid", valid_options=",".join(SUPPORTED_BACKBONES), automl_enabled="TRUE", description="Type of backbone")
    custom_args: Optional[Dict[Any, Any]] = DICT_FIELD(None, default_value=None, description="custom backbone config")
    freeze: bool = BOOL_FIELD(value=False, default_value=False, description="Flag to freeze backbone weights during training")
    pretrained: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to pretrained file")
    init_cfg: Optional[InitCfg] = None


@dataclass
class TrainAugCfg:
    """Arguments for Train Config"""

    augments: Optional[List[Dict[Any, Any]]] = LIST_FIELD(None)


@dataclass
class ModelConfig:
    """Cls model config."""

    type: str = STR_FIELD(value="ImageClassifier", default_value="ImageClassifier", description="Type of model config")
    backbone: BackboneConfig = DATACLASS_FIELD(BackboneConfig())
    neck: Optional[Dict[Any, Any]] = DICT_FIELD(None, description="Neck configuration")
    head: HeadConfig = DATACLASS_FIELD(HeadConfig())
    init_cfg: InitCfg = DATACLASS_FIELD(InitCfg())  # No change
    train_cfg: TrainAugCfg = DATACLASS_FIELD(TrainAugCfg())


@dataclass
class GenTrtEngineExpConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Results directory", display_name="Results directory")
    gpu_id: int = INT_FIELD(value=0, default_value=0, description="GPU ID", display_name="GPU ID", value_min=0)
    onnx_file: Optional[str] = STR_FIELD(value=None, default_value="", description="ONNX file", display_name="ONNX file")
    trt_engine: Optional[str] = STR_FIELD(value=None, display_name="engine file path")
    input_channel: int = INT_FIELD(value=3, valid_min=1, valid_max=3, description="Input channel", display_name="Input channel")
    input_width: int = INT_FIELD(value=224, valid_min=32, valid_max="inf", description="Input width", display_name="Input width")
    input_height: int = INT_FIELD(value=224, valid_min=32, valid_max="inf", description="Input height", display_name="Input height")
    opset_version: int = INT_FIELD(value=12, default_value=12, valid_min=11, valid_max=17, display_name="opset version", description="""Operator set version of the ONNX model used to generate the TensorRT engine.""")
    batch_size: int = INT_FIELD(value=-1, description="Batch size", display_name="Batch Size")
    verbose: bool = BOOL_FIELD(value=False, description="Verbose flag")
    tensorrt: TrtConfig = DATACLASS_FIELD(TrtConfig())


@dataclass
class ExperimentConfig:
    """Experiment config."""

    model: ModelConfig = DATACLASS_FIELD(ModelConfig())
    dataset: DatasetConfig = DATACLASS_FIELD(DatasetConfig())
    train: TrainExpConfig = DATACLASS_FIELD(TrainExpConfig())
    evaluate: EvalExpConfig = DATACLASS_FIELD(EvalExpConfig())
    inference: InferenceExpConfig = DATACLASS_FIELD(InferenceExpConfig())
    gen_trt_engine: GenTrtEngineExpConfig = DATACLASS_FIELD(GenTrtEngineExpConfig())
    export: ExportExpConfig = DATACLASS_FIELD(ExportExpConfig())
    results_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Results directory", display_name="Results directory")
