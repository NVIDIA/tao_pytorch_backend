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

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class NormConfig:
    """Configuration parameters for Normalization Preprocessing."""

    type: str = "SyncBN"  # Can be BN or SyncBN
    requires_grad: bool = True  # Whether to train the gamma beta parameters of BN


@dataclass
class TestModelConfig:
    """Configuration parameters for Inference."""

    mode: str = "whole"
    # crop_size: Optional[List[int]] = None  # Configurable
    # stride: Optional[List[int]] = None  # Configurable


@dataclass
class LossDecodeConfig:
    """Configuration parameters for Loss."""

    type: str = "CrossEntropyLoss"
    use_sigmoid: bool = False
    loss_weight: float = 1.0


@dataclass
class SegformerHeadConfig:
    """Configuration parameters for Segformer Head."""

    type: str = "TAOSegFormerHead"
    in_channels: List[int] = field(default_factory=lambda: [64, 128, 320, 512])  # [64, 128, 320, 512], [32, 64, 160, 256]
    in_index: List[int] = field(default_factory=lambda: [0, 1, 2, 3])  # No change
    feature_strides: List[int] = field(default_factory=lambda: [4, 8, 16, 32])  # No change
    # @sean this for mmseg is what decoder_params:embed_dim is to us...
    channels: int = 128  # No change
    dropout_ratio: float = 0.1
    num_classes: int = 150
    norm_cfg: NormConfig = NormConfig()
    align_corners: bool = False
    decoder_params: Dict[str, int] = field(default_factory=lambda: {"embed_dim": 768})  # 256, 512, 768 -> Configurable
    loss_decode: LossDecodeConfig = LossDecodeConfig()  # Non-configurable since there is only one loss


@dataclass
class BackboneConfig:
    """Configuration parameters for Backbone."""

    type: str = "mit_b1"
    init_cfg: Dict[str, Any] = field(default_factory=lambda: {"type": "Pretrained", "checkpoint": None})


@dataclass
class SFModelConfig:
    """SF model config."""

    type: str = "EncoderDecoder"
    pretrained_model_path: Optional[str] = None
    backbone: BackboneConfig = BackboneConfig()
    decode_head: SegformerHeadConfig = SegformerHeadConfig()
    # @sean setting this triggers an error in mmseg, report this bug
    # Basically this one should be generalized https://github.com/open-mmlab/mmsegmentation/issues/3011
    test_cfg: TestModelConfig = TestModelConfig()
    input_width: int = 512
    input_height: int = 512


# Use the field parameter in order to define as dictionaries
@dataclass
class RandomCropCfg:
    """Configuration parameters for Random Crop Aug."""

    crop_size: List[int] = field(default_factory=lambda: [512, 512])  # Non - configurable
    cat_max_ratio: float = 0.75


@dataclass
class ResizeCfg:
    """Configuration parameters for Resize Preprocessing."""

    img_scale: Optional[List[int]] = None  # configurable
    ratio_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
    keep_ratio: bool = True


@dataclass
class SFAugmentationConfig:
    """Augmentation config."""

    # @subha: TO Do: Add some more augmentation configurations which were not used in Segformer (later)
    random_crop: RandomCropCfg = RandomCropCfg()
    resize: ResizeCfg = ResizeCfg()
    random_flip: Dict[str, float] = field(default_factory=lambda: {'prob': 0.5})
    color_aug: Dict[str, str] = field(default_factory=lambda: {'type': 'PhotoMetricDistortion'})


@dataclass
class ImgNormConfig:
    """Configuration parameters for Img Normalization."""

    mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])
    std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])
    to_rgb: bool = True
    pad_val: int = 0
    seg_pad_val: int = 255
    type: str = "SegDataPreProcessor"


@dataclass
class PipelineConfig:
    """Configuration parameters for Validation Pipe."""

    img_norm_cfg: ImgNormConfig = ImgNormConfig()
    multi_scale: Optional[List[int]] = None
    augmentation_config: SFAugmentationConfig = SFAugmentationConfig()
    Pad: Dict[str, int] = field(default_factory=lambda: {'size_ht': 1024, 'size_wd': 1024, 'pad_val': 0, 'seg_pad_val': 255})  # Non-configurable. Set based on model_input
    CollectKeys: List[str] = field(default_factory=lambda: ['img', 'gt_semantic_seg'])


@dataclass
class seg_class:
    """Indiv color."""

    seg_class: str = "background"
    mapping_class: str = "background"
    label_id: int = 0
    rgb: List[int] = field(default_factory=lambda: [255, 255, 255])


@dataclass
class SFDatasetConfig:
    """Dataset Config."""

    img_dir: Optional[Any] = None
    ann_dir: Optional[Any] = None
    pipeline: PipelineConfig = PipelineConfig()


@dataclass
class SFDatasetExpConfig:
    """Dataset config."""

    data_root: Optional[str] = None
    img_norm_cfg: ImgNormConfig = ImgNormConfig()
    train_dataset: SFDatasetConfig = SFDatasetConfig()
    val_dataset: SFDatasetConfig = SFDatasetConfig()
    test_dataset: SFDatasetConfig = SFDatasetConfig()
    palette: Optional[List[seg_class]] = None
    seg_class_default: seg_class = seg_class()
    dataloader: str = "Dataloader"
    img_suffix: Optional[str] = ".png"
    seg_map_suffix: Optional[str] = ".png"
    repeat_data_times: int = 2
    batch_size: int = 32
    workers_per_gpu: int = 8
    shuffle: bool = True
    input_type: str = "rgb"

    # This is set to false because we only have 2 classes in the example
    reduce_zero_label: bool = False
    type: str = "BaseSegDataset"


@dataclass
class SFEnvConfig:
    """ Env Config for Segformer. """

    cudnn_benchmark: bool = True
    mp_cfg: Dict[Any, Any] = field(default_factory=lambda: {'mp_start_method': 'fork', 'opencv_num_threads': 0})
    dist_cfg: Dict[str, str] = field(default_factory=lambda: {'backend': 'nccl'})


@dataclass
class SFExpConfig:
    """ Overall Exp Config for Segformer. """

    manual_seed: int = 47
    distributed: bool = True
    # If needed, the next line can be commented
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    MASTER_ADDR: str = "127.0.0.1"
    MASTER_PORT: int = 25678

    deterministic: bool = False
    env_cfg: SFEnvConfig = SFEnvConfig()
    default_scope: str = "mmseg"
    log_level: str = "INFO"


@dataclass
class MultiStepLRConfig:
    """Configuration parameters for Multi Step Optimizer."""

    lr_steps: List[int] = field(default_factory=lambda: [15, 25])
    lr_decay: float = 0.1


# This abides by API spec but is translated to mmengine via LinearLR/PolyLRConfig
@dataclass
class LRConfig:
    """Configuration parameters for LR Scheduler."""

    # Check what is _delete_ is
    policy: str = "poly"  # Non-configurable
    warmup: str = 'linear'  # Non-configurable
    warmup_iters: int = 1500
    warmup_ratio: float = 1e-6
    power: float = 1.0
    min_lr: float = 0.0
    by_epoch: bool = False


@dataclass
class LinearLRConfig:
    """Configuration parameters for Linear LR."""

    type: str = "LinearLR"
    start_factor: float = 1e-6
    by_epoch: bool = False
    begin: int = 0
    end: int = 1500


@dataclass
class PolyLRConfig:
    """Configuration parameters for Polynomial LR decay."""

    type: str = 'PolyLR'
    eta_min: float = 0.0
    power: float = 1.0
    begin: int = 1500
    end: int = 160000
    by_epoch: bool = False


@dataclass
class ParamwiseConfig:
    """Configuration parameters for Parameters."""

    pos_block: Dict[str, float] = field(default_factory=lambda: {"decay_mult": 0.0})
    norm: Dict[str, float] = field(default_factory=lambda: {"decay_mult": 0.0})
    head: Dict[str, float] = field(default_factory=lambda: {"lr_mult": 10.0})


@dataclass
class SFOptimConfig:
    """Optimizer config."""

    type: str = "AdamW"
    lr: float = 0.00006
    betas: List[float] = field(default_factory=lambda: [0.0, 0.9])
    weight_decay: float = 5e-4
    paramwise_cfg: ParamwiseConfig = ParamwiseConfig()


@dataclass
class TrainerConfig:
    """Train Config."""

    sf_optim: SFOptimConfig = SFOptimConfig()
    lr_config: LRConfig = LRConfig()
    grad_clip: float = 0.0
    find_unused_parameters: bool = True


@dataclass
class SFTrainExpConfig:
    """Train experiment config."""

    results_dir: Optional[str] = None
    encryption_key: Optional[str] = None
    exp_config: SFExpConfig = SFExpConfig()
    trainer: TrainerConfig = TrainerConfig()
    num_gpus: int = 1  # non configurable here
    max_iters: int = 10
    logging_interval: int = 1
    checkpoint_interval: int = 1
    resume_training_checkpoint_path: Optional[str] = None
    validation_interval: Optional[int] = 1
    validate: bool = True

    param_scheduler: List[Any] = field(default_factory=lambda: [LinearLRConfig(), PolyLRConfig()])
    resume: bool = False

    default_hooks: Dict[Any, Any] = field(default_factory=lambda: {'timer': {'type': 'IterTimerHook'},
                                                                   'logger': {'type': 'TAOTextLoggerHook', 'interval': 1, 'log_metric_by_epoch': False},
                                                                   'param_scheduler': {'type': 'ParamSchedulerHook'},
                                                                   'checkpoint': {'type': 'CheckpointHook', 'by_epoch': False, 'interval': 1},
                                                                   'sampler_seed': {'type': 'DistSamplerSeedHook'},
                                                                   'visualization': {'type': 'SegVisualizationHook'}})


@dataclass
class SFInferenceExpConfig:
    """Inference experiment config."""

    encryption_key: Optional[str] = None
    results_dir: Optional[str] = None
    gpu_id: int = 0
    checkpoint: Optional[str] = None
    exp_config: SFExpConfig = SFExpConfig()
    num_gpus: int = 1  # non configurable here
    trt_engine: Optional[str] = None


@dataclass
class SFEvalExpConfig:
    """Inference experiment config."""

    results_dir: Optional[str] = None
    encryption_key: Optional[str] = None
    gpu_id: int = 0
    checkpoint: Optional[str] = None
    exp_config: SFExpConfig = SFExpConfig()
    num_gpus: int = 1  # non configurable here
    trt_engine: Optional[str] = None


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = "FP32"
    workspace_size: int = 1024
    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 1


@dataclass
class OnnxConfig:
    """Onnx config."""

    type: str = 'onnx'
    export_params: bool = True
    keep_initializers_as_inputs: bool = False
    opset_version: int = 13
    save_file: Optional[str] = None
    input_names: Optional[Any] = None
    output_names: Optional[Any] = None
    input_shape: Optional[Any] = None
    optimize: bool = True


@dataclass
class CodebaseConfig:
    """Codebase config for onnx"""

    type: str = 'mmseg'
    task: str = 'Segmentation'
    with_argmax: bool = True


@dataclass
class SFExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = None
    encryption_key: Optional[str] = None
    verify: bool = True
    simplify: bool = False
    batch_size: int = 1
    opset_version: int = 11
    trt_engine: Optional[str] = None
    checkpoint: Optional[str] = None
    onnx_file: Optional[str] = None
    exp_config: SFExpConfig = SFExpConfig()
    trt_config: TrtConfig = TrtConfig()
    num_gpus: int = 1  # non configurable here
    input_channel: int = 3
    input_width: int = 1024
    input_height: int = 1024

    onnx_config: OnnxConfig = OnnxConfig()
    codebase_config: CodebaseConfig = CodebaseConfig()


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

    model: SFModelConfig = SFModelConfig()
    dataset: SFDatasetExpConfig = SFDatasetExpConfig()
    train: SFTrainExpConfig = SFTrainExpConfig()
    evaluate: SFEvalExpConfig = SFEvalExpConfig()
    inference: SFInferenceExpConfig = SFInferenceExpConfig()
    gen_trt_engine: GenTrtEngineExpConfig = GenTrtEngineExpConfig()
    export: SFExportExpConfig = SFExportExpConfig()
    encryption_key: Optional[str] = None
    results_dir: str = MISSING
