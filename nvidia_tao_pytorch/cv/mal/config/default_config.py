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
from dataclasses import dataclass

from nvidia_tao_pytorch.core.common_config import EvaluateConfig, CommonExperimentConfig, InferenceConfig, TrainConfig
from nvidia_tao_pytorch.config.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    FLOAT_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class MALInferenceExpConfig(InferenceConfig):
    """Inference configuration template."""

    ann_path: str = STR_FIELD(value="/data/raw-data/annotations/instances_val2017.json")
    img_dir: str = STR_FIELD(value="/data/raw-data/val2017/")
    label_dump_path: str = STR_FIELD(value="instances_val2017_mal.json")
    batch_size: int = INT_FIELD(value=3, default_value=3, valid_min=1, valid_max="inf")
    load_mask: bool = BOOL_FIELD(value=False)


@dataclass
class MALEvalExpConfig(EvaluateConfig):
    """Evaluation configuration template."""

    batch_size: int = INT_FIELD(value=3, default_value=3, valid_min=1, valid_max="inf")
    use_mixed_model_test: bool = BOOL_FIELD(value=False)
    use_teacher_test: bool = BOOL_FIELD(value=False)
    comp_clustering: bool = BOOL_FIELD(value=False)
    use_flip_test: bool = BOOL_FIELD(value=False)


@dataclass
class MALDatasetConfig:
    """Data configuration template."""

    type: str = STR_FIELD(value='coco', default_value="coco", valid_options="coco", display_name="dataset type")
    train_ann_path: str = STR_FIELD(value='/data/raw-data/annotations/instances_train2017.json', display_name="Annotation path of the training set")
    train_img_dir: str = STR_FIELD(value='/data/raw-data/train2017/', display_name="Image directory of the training set")
    val_ann_path: str = STR_FIELD(value='/data/raw-data/annotations/instances_val2017.json', display_name="Annotation path of the validation set")
    val_img_dir: str = STR_FIELD(value='/data/raw-data/val2017/', display_name="Image directory of the validation set")
    min_obj_size: float = FLOAT_FIELD(value=2048, default_value=2048, display_name="minimum object size")
    max_obj_size: float = FLOAT_FIELD(value=1e10, default_value="1.00E+10", display_name="maximum object size")
    num_workers_per_gpu: int = INT_FIELD(value=2, default_value=2)
    load_mask: bool = BOOL_FIELD(value=True, display_name="Whether to load segmentation mask in annotation file")
    crop_size: int = INT_FIELD(value=512, default_value=512, valid_min=256, valid_max="inf")


@dataclass
class MALModelConfig:
    """Model configuration template."""

    arch: str = STR_FIELD(value='vit-mae-base/16', value_type="ordered", default_value="vit-mae-base/16", valid_options="vit-deit-tiny/16,vit-deit-small/16,vit-mae-base/16,vit-mae-large/16,vit-mae-huge/14,fan_tiny_12_p16_224,fan_small_12_p16_224,fan_base_18_p16_224,fan_large_24_p16_224,fan_tiny_8_p4_hybrid,fan_small_12_p4_hybrid,fan_base_16_p4_hybrid,fan_large_16_p4_hybrid")
    frozen_stages: List[int] = LIST_FIELD(arrList=[-1], value_type="list_1_backbone", default_value=[-1])
    mask_head_num_convs: int = INT_FIELD(value=4, default_value=4, valid_min=1, valid_max="inf")
    mask_head_hidden_channel: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf")
    mask_head_out_channel: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf")
    teacher_momentum: float = FLOAT_FIELD(value=0.996, default_value=0.996, valid_min=0, valid_max=1)
    not_adjust_scale: bool = BOOL_FIELD(value=False)
    mask_scale_ratio_pre: int = INT_FIELD(value=1)
    mask_scale_ratio: float = FLOAT_FIELD(value=2.0)
    vit_dpr: float = FLOAT_FIELD(value=0)


@dataclass
class MALTrainExpConfig(TrainConfig):
    """Train configuration template."""

    batch_size: int = INT_FIELD(value=3, default_value=1, valid_min=1, valid_max="inf")
    accum_grad_batches: int = INT_FIELD(value=1, default_value=1, valid_min=1, valid_max="inf")
    use_amp: bool = BOOL_FIELD(value=True)
    pretrained_model_path: Optional[str] = STR_FIELD(value=None, default_value="")

    # optim
    optim_type: str = STR_FIELD(value='adamw', valid_options="adamw")
    optim_momentum: float = FLOAT_FIELD(value=0.9, default_value=0.9, valid_min=0, valid_max=1)
    lr: float = FLOAT_FIELD(value=0.000001, default_value=0.000001, valid_min=0, valid_max="inf")
    min_lr: float = FLOAT_FIELD(value=0)
    min_lr_rate: float = FLOAT_FIELD(value=0.2, default_value=0.02, valid_min=0, valid_max=1)
    num_wave: float = FLOAT_FIELD(value=1)
    wd: float = FLOAT_FIELD(value=0.0005)
    optim_eps: float = FLOAT_FIELD(value=1e-8)
    optim_betas: List[float] = LIST_FIELD([0.9, 0.9])
    warmup_epochs: int = INT_FIELD(value=1, default_value=1, valid_min=0, valid_max="inf")

    margin_rate: List[float] = LIST_FIELD([0, 1.2])
    test_margin_rate: List[float] = LIST_FIELD([0.6, 0.6])
    mask_thres: List[float] = LIST_FIELD([0.1])

    # loss
    loss_mil_weight: float = FLOAT_FIELD(value=4)
    loss_crf_weight: float = FLOAT_FIELD(value=0.5)

    # crf
    crf_zeta: float = FLOAT_FIELD(value=0.1)
    crf_kernel_size: int = INT_FIELD(value=3)
    crf_num_iter: int = INT_FIELD(value=100)
    loss_crf_step: int = INT_FIELD(value=4000)
    loss_mil_step: int = INT_FIELD(value=1000)
    crf_size_ratio: int = INT_FIELD(value=1)
    crf_value_high_thres: float = FLOAT_FIELD(value=0.9)
    crf_value_low_thres: float = FLOAT_FIELD(value=0.1)


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment configuration template."""

    strategy: str = STR_FIELD(value='ddp')
    dataset: MALDatasetConfig = DATACLASS_FIELD(MALDatasetConfig())
    train: MALTrainExpConfig = DATACLASS_FIELD(MALTrainExpConfig())
    model: MALModelConfig = DATACLASS_FIELD(MALModelConfig())
    inference: MALInferenceExpConfig = DATACLASS_FIELD(MALInferenceExpConfig())
    evaluate: MALEvalExpConfig = DATACLASS_FIELD(MALEvalExpConfig())
