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
class CNOptimConfig:
    """Optimizer config."""

    monitor_name: str = "val_loss"
    optim: str = "adamw"
    lr: float = 0.0001
    policy: str = "linear"
    momentum: float = 0.9
    weight_decay: float = 0.01


@dataclass
class ChangeNetHeadConfig:
    """Configuration parameters for Visual ChangeNet Head."""

    in_channels: List[int] = field(default_factory=lambda: [128, 256, 384, 384])  # FANHybrid-S
    in_index: List[int] = field(default_factory=lambda: [0, 1, 2, 3])  # No change
    feature_strides: List[int] = field(default_factory=lambda: [4, 8, 16, 16])
    align_corners: bool = False
    decoder_params: Dict[str, int] = field(default_factory=lambda: {"embed_dim": 256})  # 256, 512, 768 -> Configurable


@dataclass
class BackboneConfig:
    """Configuration parameters for Backbone."""

    type: str = "fan_small_12_p4_hybrid"
    feat_downsample: bool = False
    pretrained_backbone_path: Optional[str] = None
    freeze_backbone: bool = False


@dataclass
class CNModelClassifyConfig:
    """CN Model Classification config."""

    train_margin_euclid: float = 2.0
    eval_margin: float = 2.0
    embedding_vectors: int = 5
    embed_dec: int = 5
    learnable_difference_modules: int = 4
    difference_module: Optional[str] = 'learnable'


@dataclass
class CNModelConfig:
    """CN Model config."""

    backbone: BackboneConfig = BackboneConfig()
    decode_head: ChangeNetHeadConfig = ChangeNetHeadConfig()
    classify: CNModelClassifyConfig = CNModelClassifyConfig()


@dataclass
class RandomFlip:
    """RandomFlip augmentation config."""

    vflip_probability: float = 0.5
    hflip_probability: float = 0.5
    enable: bool = True


@dataclass
class RandomRotation:
    """RandomRotation augmentation config."""

    rotate_probability: float = 0.5
    angle_list: List[float] = field(default_factory=lambda: [90, 180, 270])
    enable: bool = True


@dataclass
class RandomColor:
    """RandomColor augmentation config."""

    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    hue: float = 0.3
    enable: bool = True


@dataclass
class RandomCropWithScale:
    """RandomCropWithScale augmentation config."""

    scale_range: List[float] = field(default_factory=lambda: [1, 1.2])  # non configurable here
    enable: bool = True


@dataclass
class CNAugmentationSegmentConfig:
    """Augmentation config for segmentation."""

    random_flip: RandomFlip = RandomFlip()
    random_rotate: RandomRotation = RandomRotation()
    random_color: RandomColor = RandomColor()
    with_scale_random_crop: RandomCropWithScale = RandomCropWithScale()
    with_random_blur: bool = True
    with_random_crop: bool = True
    mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])  # non configurable here
    std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])  # non configurable here


@dataclass
class CNAugmentationClassifyConfig:
    """Augmentation config for classification."""

    rgb_input_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    rgb_input_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class DataPathFormat:
    """Dataset Path experiment config."""

    csv_path: str = MISSING
    images_dir: str = MISSING


@dataclass
class CNDatasetClassifyConfig:
    """Classification Dataset Config."""

    train_dataset: DataPathFormat = DataPathFormat()
    validation_dataset: DataPathFormat = DataPathFormat()
    test_dataset: DataPathFormat = DataPathFormat()
    infer_dataset: DataPathFormat = DataPathFormat()
    image_ext: Optional[str] = None
    batch_size: int = 32
    workers: int = 8
    fpratio_sampling: float = 0.1
    num_input: int = 8
    input_map: Optional[Dict[str, int]] = None
    grid_map: Optional[Dict[str, int]] = None
    concat_type: Optional[str] = None
    output_shape: List[int] = field(default_factory=lambda: [100, 100])
    augmentation_config: CNAugmentationClassifyConfig = CNAugmentationClassifyConfig()
    num_classes: int = 2


@dataclass
class CNDatasetSegmentConfig:
    """Segmentation Dataset Config."""

    root_dir: str = MISSING
    label_transform: str = "norm"
    data_name: str = "LEVIR"
    dataset: str = "CNDataset"
    multi_scale_train: bool = True
    multi_scale_infer: bool = False
    num_classes: int = 2
    img_size: int = 256
    batch_size: int = 8
    workers: int = 2
    shuffle: bool = True
    image_folder_name: str = "A"
    change_image_folder_name: str = 'B'
    list_folder_name: str = 'list'
    annotation_folder_name: str = "label"
    augmentation: CNAugmentationSegmentConfig = CNAugmentationSegmentConfig()
    train_split: str = 'train'
    validation_split: str = 'val'
    test_split: str = 'test'
    predict_split: str = 'test'
    label_suffix: str = '.png'
    color_map: Optional[Dict[str, List[int]]] = None


@dataclass
class CNDatasetConfig:
    """Dataset config."""

    segment: CNDatasetSegmentConfig = CNDatasetSegmentConfig()
    classify: CNDatasetClassifyConfig = CNDatasetClassifyConfig()


@dataclass
class TensorBoardLogger:
    """Configuration for the tensorboard logger."""

    enabled: bool = False
    infrequent_logging_frequency: int = 2  # Defined per epoch


@dataclass
class CNTrainClassifyConfig:
    """Classifier loss config."""

    loss: str = "ce"  # ce, contrastive
    cls_weight: List[float] = field(default_factory=lambda: [1.0, 10.0])


@dataclass
class CNTrainSegmentConfig:
    """Segmentation loss Config."""

    loss: str = 'ce'
    weights: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.8, 1.0])


@dataclass
class CNTrainConfig:
    """Train Config."""

    optim: CNOptimConfig = CNOptimConfig()
    num_epochs: int = 200
    num_nodes: int = 1
    val_interval: int = 1
    checkpoint_interval: int = 1
    pretrained_model_path: Optional[str] = None
    resume_training_checkpoint_path: Optional[str] = None
    results_dir: Optional[str] = None
    classify: CNTrainClassifyConfig = CNTrainClassifyConfig()
    segment: CNTrainSegmentConfig = CNTrainSegmentConfig()  # non configurable here
    tensorboard: Optional[TensorBoardLogger] = TensorBoardLogger()


@dataclass
class CNEvalExpConfig:
    """Evaluation experiment config."""

    num_gpus: int = 1  # non configurable here
    checkpoint: Optional[str] = None
    results_dir: Optional[str] = None
    vis_after_n_batches: int = 16
    trt_engine: str = MISSING
    batch_size: int = -1


@dataclass
class CNInferenceExpConfig:
    """Inference experiment config."""

    num_gpus: int = 1  # non configurable here
    checkpoint: Optional[str] = None
    results_dir: Optional[str] = None
    vis_after_n_batches: int = 1
    trt_engine: str = MISSING
    batch_size: int = -1
    gpu_id: int = 0


@dataclass
class CNExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = None
    gpu_id: int = 0
    checkpoint: str = MISSING
    onnx_file: str = MISSING
    on_cpu: bool = False
    input_channel: int = 3
    input_width: int = 256
    input_height: int = 256
    opset_version: int = 12
    batch_size: int = -1
    verbose: bool = False


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
class CNGenTrtEngineExpConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = None
    gpu_id: int = 0
    onnx_file: str = MISSING
    trt_engine: Optional[str] = None
    input_channel: int = 3
    input_width: int = 256
    input_height: int = 256
    opset_version: int = 12
    batch_size: int = -1
    verbose: bool = False
    tensorrt: TrtConfig = TrtConfig()


@dataclass
class ExperimentConfig:
    """Experiment config."""

    model: CNModelConfig = CNModelConfig()
    dataset: CNDatasetConfig = CNDatasetConfig()
    train: CNTrainConfig = CNTrainConfig()
    evaluate: CNEvalExpConfig = CNEvalExpConfig()
    inference: CNInferenceExpConfig = CNInferenceExpConfig()
    export: CNExportExpConfig = CNExportExpConfig()
    gen_trt_engine: CNGenTrtEngineExpConfig = CNGenTrtEngineExpConfig()
    encryption_key: Optional[str] = None
    results_dir: str = MISSING
    num_gpus: int = 1  # non configurable here
    task: Optional[str] = 'segment'
