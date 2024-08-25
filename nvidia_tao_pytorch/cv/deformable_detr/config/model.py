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

"""Configuration hyperparameter schema for the model."""

from dataclasses import dataclass
from typing import List, Optional

from nvidia_tao_pytorch.config.types import (
    BOOL_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)
from nvidia_tao_pytorch.cv.deformable_detr.model.gc_vit import gc_vit_model_dict

SUPPORTED_BACKBONES = [
    *list(gc_vit_model_dict.keys()),
    *["resnet_50"],
]


@dataclass
class DDModelConfig:
    """Deformable DETR model config."""

    pretrained_backbone_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="pretrained backbone path",
        description="[Optional] Path to a pretrained backbone file.",
    )
    backbone: str = STR_FIELD(
        value="resnet_50",
        default_value="resnet_50",
        display_name="backbone",
        description="""The backbone name of the model.
                    TAO implementation of Deformable DETR support GCViT and ResNet50.""",
        valid_options=",".join(SUPPORTED_BACKBONES)
    )
    num_queries: int = INT_FIELD(
        value=300,
        default_value=300,
        description="The number of queries",
        display_name="number of queries",
        valid_min=1,
        valid_max="inf",
        automl_enabled="TRUE"
    )
    num_feature_levels: int = INT_FIELD(
        value=4,
        default_value=4,
        description="The number of feature levels to use in the model",
        display_name="number of feature levels",
        valid_min=1,
        valid_max=5,
    )
    cls_loss_coef: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the classification error in the matching cost.",
        display_name="Class loss coefficient",
    )
    bbox_loss_coef: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the L1 error of the bounding box coordinates in the matching cost.",
        display_name="BBox loss coefficient",
    )
    giou_loss_coef: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the GIoU loss of the bounding box in the matching cost.",
        display_name="GIoU loss coefficient",
    )
    with_box_refine: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="With box refine",
        description="""A flag specifying whether to enbable the Iterative Bounding Box Refinement""",
    )
    num_select: int = INT_FIELD(
        value=300,
        default_value=300,
        description="The number of top-K predictions selected during post-process",
        display_name="num select",
        valid_min=1,
        automl_enabled="TRUE"
    )

    return_interm_indices: List[int] = LIST_FIELD(
        arrList=[1, 2, 3, 4],
        description="The index of feature levels to use in the model. The length must match `num_feature_levels`.",
        display_name="return interim indices"
    )

    focal_alpha: float = FLOAT_FIELD(
        value=0.25,
        description="The alpha value in the focal loss.",
        display_name="focal alpha",
        math_cond="> 0.0"
    )
    clip_max_norm: float = FLOAT_FIELD(
        value=0.1,
        display_name="clip max norm",
        description="",
    )
    nheads: int = INT_FIELD(
        value=8,
        default_value=8,
        description="Number of heads",
        display_name="nheads",
    )
    dropout_ratio: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        description="The probability to drop hidden units.",
        display_name="drop out ratio",
        valid_min=0.0,
        valid_max=1.0
    )
    hidden_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Dimension of the hidden units.",
        display_unit="hidden dim",
        automl_enabled="FALSE"
    )
    enc_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        description="Numer of encoder layers in the transformer",
        valid_min=1,
        automl_enabled="TRUE",
        display_name="encoder layers",
    )
    dec_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        description="Numer of decoder layers in the transformer",
        valid_min=1,
        automl_enabled="TRUE",
        display_name="decoder layers",
    )
    dim_feedforward: int = INT_FIELD(
        value=1024,
        description="Dimension of the feedforward network",
        display_name="dim feedforward",
        valid_min=1,
    )
    dec_n_points: int = INT_FIELD(
        value=4,
        display_name="decoder n points",
        description="Number of reference points in the decoder.",
        valid_min=1,
    )
    enc_n_points: int = INT_FIELD(
        value=4,
        display_name="encoder n points",
        description="Number of reference points in the encoder.",
        valid_min=1,
    )
    aux_loss: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="Train backbone",
        description="""A flag specifying whether to use auxiliary
                    decoding losses (loss at each decoder layer)""",
    )
    dilation: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Dilation enabled.",
        description="""A flag specifying whether enable dilation or not in the backbone.""",
    )
    train_backbone: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="Train backbone",
        description="""Flag to set backbone weights as trainable or frozen.
                    When set to `False`, the backbone weights will be frozen.""",
    )
    loss_types: List[str] = LIST_FIELD(
        arrList=['labels', 'boxes'],
        description="Losses to be used during training",
        display_name="loss_types",
    )
    backbone_names: List[str] = LIST_FIELD(
        arrList=["backbone.0"],
        description="Prefix of the tensor names corresponding to the backbone.",
        display_name="Backbone tensor name prefix")
    linear_proj_names: List[str] = LIST_FIELD(
        arrList=['reference_points', 'sampling_offsets'],
        display_name="linear projection names",
        description="Linear projection layer names."
    )
