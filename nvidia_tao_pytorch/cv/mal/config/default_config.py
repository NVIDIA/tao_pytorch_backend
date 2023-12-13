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

from typing import List, Optional
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class InferConfig:
    """Inference configuration template."""

    ann_path: str = MISSING
    img_dir: str = MISSING
    label_dump_path: str = MISSING
    batch_size: int = 3
    load_mask: bool = False
    results_dir: Optional[str] = None


@dataclass
class EvalConfig:
    """Evaluation configuration template."""

    batch_size: int = 3
    use_mixed_model_test: bool = False
    use_teacher_test: bool = False
    comp_clustering: bool = False
    use_flip_test: bool = False
    results_dir: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration template."""

    type: str = 'coco'
    train_ann_path: str = ''
    train_img_dir: str = ''
    val_ann_path: str = ''
    val_img_dir: str = ''
    min_obj_size: float = 2048
    max_obj_size: float = 1e10
    num_workers_per_gpu: int = 2
    load_mask: bool = True
    crop_size: int = 512


@dataclass
class ModelConfig:
    """Model configuration template."""

    arch: str = 'vit-mae-base/16'
    frozen_stages: List[int] = field(default_factory=lambda: [-1])
    mask_head_num_convs: int = 4
    mask_head_hidden_channel: int = 256
    mask_head_out_channel: int = 256
    teacher_momentum: float = 0.996
    not_adjust_scale: bool = False
    mask_scale_ratio_pre: int = 1
    mask_scale_ratio: float = 2.0
    vit_dpr: float = 0


@dataclass
class TrainConfig:
    """Train configuration template."""

    seed: int = 1
    num_epochs: int = 10
    save_every_k_epoch: int = 1
    val_interval: int = 1
    batch_size: int = 3
    accum_grad_batches: int = 1
    use_amp: bool = True

    # optim
    optim_type: str = 'adamw'
    optim_momentum: float = 0.9
    lr: float = 0.000001
    min_lr: float = 0
    min_lr_rate: float = 0.2
    num_wave: float = 1
    wd: float = 0.0005
    optim_eps: float = 1e-8
    optim_betas: List[float] = field(default_factory=lambda: [0.9, 0.9])
    warmup_epochs: int = 1

    margin_rate: List[float] = field(default_factory=lambda: [0, 1.2])
    test_margin_rate: List[float] = field(default_factory=lambda: [0.6, 0.6])
    mask_thres: List[float] = field(default_factory=lambda: [0.1])

    # loss
    loss_mil_weight: float = 4
    loss_crf_weight: float = 0.5

    # crf
    crf_zeta: float = 0.1
    crf_kernel_size: int = 3
    crf_num_iter: int = 100
    loss_crf_step: int = 4000
    loss_mil_step: int = 1000
    crf_size_ratio: int = 1
    crf_value_high_thres: float = 0.9
    crf_value_low_thres: float = 0.1
    results_dir: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    gpu_ids: List[int] = field(default_factory=lambda: [])
    strategy: str = 'ddp'
    num_nodes: int = 1
    checkpoint: Optional[str] = None
    dataset: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()
    inference: InferConfig = InferConfig()
    evaluate: EvalConfig = EvalConfig()
    results_dir: str = MISSING
