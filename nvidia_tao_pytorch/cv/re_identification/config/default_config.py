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


@dataclass
class ReIDModelConfig:
    """Re-Identification model configuration for training, testing & validation."""

    backbone: str = "resnet_50"
    last_stride: int = 1
    pretrain_choice: str = "imagenet"
    pretrained_model_path: Optional[str] = None
    input_channels: int = 3
    input_width: int = 128
    input_height: int = 256
    neck: str = "bnneck"
    feat_dim: int = 256
    neck_feat: str = "after"
    metric_loss_type: str = "triplet"
    with_center_loss: bool = False
    with_flip_feature: bool = False
    label_smooth: bool = True
    pretrain_hw_ratio: float = 2
    id_loss_type: str = "softmax"
    id_loss_weight: float = 1.0
    triplet_loss_weight: float = 1.0
    no_margin: bool = False
    cos_layer: bool = False
    dropout_rate: float = 0.0
    reduce_feat_dim: bool = False
    drop_path: float = 0.1
    drop_out: float = 0.0
    att_drop_rate: float = 0.0
    stride_size:  List[int] = field(default_factory=lambda: [16, 16])
    gem_pooling: bool = False
    stem_conv: bool = False
    jpm: bool = False
    shift_num: int = 5
    shuffle_group: int = 2
    devide_length: int = 4
    re_arrange: bool = True
    sie_coe: float = 3.0
    sie_camera: bool = False
    sie_view: bool = False
    semantic_weight: float = 1.0


@dataclass
class OptimConfig:
    """Optimizer configuration for the LR scheduler."""

    name: str = "Adam"
    lr_monitor: str = "val_loss"
    lr_steps:  List[int] = field(default_factory=lambda: [40, 70])
    gamma: float = 0.1
    bias_lr_factor: float = 1
    weight_decay: float = 0.0005
    weight_decay_bias: float = 0.0005
    warmup_factor: float = 0.01
    warmup_iters: int = 10
    warmup_epochs: int = 20
    warmup_method: str = 'linear'
    base_lr: float = 0.00035
    momentum: float = 0.9
    center_loss_weight: float = 0.0005
    center_lr: float = 0.5
    triplet_loss_margin: float = 0.3
    large_fc_lr: bool = False
    seed: int = 1234
    cosine_margin: float = 0.5
    cosine_scale: float = 30
    trp_l2: bool = False


@dataclass
class ReIDDatasetConfig:
    """Re-Identification Dataset configuration template."""

    train_dataset_dir: Optional[str] = None
    test_dataset_dir: Optional[str] = None
    query_dataset_dir: Optional[str] = None
    num_classes: int = 751
    batch_size: int = 64
    val_batch_size: int = 128
    num_workers: int = 8
    pixel_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    pixel_std: List[float] = field(default_factory=lambda: [0.226, 0.226, 0.226])
    padding: int = 10
    prob: float = 0.5
    re_prob: float = 0.5
    sampler: str = "softmax_triplet"
    num_instances: int = 4


@dataclass
class ReIDReRankingConfig:
    """Re-Ranking configuration template for evaluation."""

    re_ranking: bool = False
    k1: int = 20
    k2: int = 6
    lambda_value: float = 0.3
    max_rank: int = 10
    num_query: int = 10


@dataclass
class ReIDTrainExpConfig:
    """Train experiment configuration template."""

    results_dir: Optional[str] = None
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_gpus: int = 1
    resume_training_checkpoint_path: Optional[str] = None
    optim: OptimConfig = OptimConfig()
    num_epochs: int = 1
    checkpoint_interval: int = 5
    grad_clip: float = 0.0


@dataclass
class ReIDInferenceExpConfig:
    """Inference experiment configuration template."""

    results_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    output_file: Optional[str] = None
    test_dataset: Optional[str] = None
    query_dataset: Optional[str] = None
    gpu_id: int = 0


@dataclass
class ReIDEvalExpConfig:
    """Evaluation experiment configuration template."""

    results_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    output_sampled_matches_plot: Optional[str] = None
    output_cmc_curve_plot: Optional[str] = None
    test_dataset: Optional[str] = None
    query_dataset: Optional[str] = None
    gpu_id: int = 0


@dataclass
class ReIDExportExpConfig:
    """Export experiment configuraiton template."""

    results_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    onnx_file: Optional[str] = None
    gpu_id: int = 0


@dataclass
class ExperimentConfig:
    """Experiment config."""

    results_dir: Optional[str] = None
    encryption_key: Optional[str] = None
    model: ReIDModelConfig = ReIDModelConfig()
    dataset: ReIDDatasetConfig = ReIDDatasetConfig()
    re_ranking: ReIDReRankingConfig = ReIDReRankingConfig()
    train: ReIDTrainExpConfig = ReIDTrainExpConfig()
    inference: ReIDInferenceExpConfig = ReIDInferenceExpConfig()
    evaluate: ReIDEvalExpConfig = ReIDEvalExpConfig()
    export: ReIDExportExpConfig = ReIDExportExpConfig()
