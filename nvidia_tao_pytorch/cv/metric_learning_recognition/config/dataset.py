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

"""Configuration hyperparameter schema for the dataset."""

from dataclasses import dataclass
from typing import Optional, List, Dict

from nvidia_tao_pytorch.config.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)


@dataclass
class GaussianBlur:
    """Gaussian Blur configuration template."""

    enabled: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to add gaussian blur to dataloader or not.",
        display_name="enabled"
    )
    kernel: List[int] = LIST_FIELD(
        arrList=[15, 15],
        description="The kernel size for the Gaussian blur.",
        display_name="kernel"
    )
    sigma: List[float] = LIST_FIELD(
        arrList=[0.3, 0.7],
        description="The sigma value range for the Gaussian blur.",
        display_name="sigma"
    )


@dataclass
class ColorAugmentation:
    """Color Augmentation configuration template."""

    enabled: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to add color augmentation to dataloader or not.",
        display_name="enabled"
    )
    brightness: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        display_name="brightness",
        description="The value of jittering brightness",
    )
    contrast: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        display_name="contrast",
        description="The value of jittering contrast",
    )
    saturation: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        display_name="saturation",
        description="The value of jittering saturation",
    )
    hue: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        display_name="hue",
        description="The value of jittering hue",
    )


@dataclass
class MLDatasetConfig:
    """Metric Learning Recognition Dataset configuration template."""

    train_dataset: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="train dataset",
        description="The path to the train dataset. This field is only required for the train task.",
    )
    val_dataset: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        default_value=None,
        description="""The map of reference set and query set addresses:
                    * reference : The directory that contains the ImageNet format reference images
                    * query : The directory that contains the ImageNet format query images""",
        display_name="validation dataset",
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=8,
        description="The number of parallel workers processing data",
        display_name="workers",
    )
    class_map: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="class_map",
        description="""[Optional] a YAML file mapping dataset class names to desired class names.
                    If not specified, by default the reported class names are the folder names
                    in the dataset folder."""
    )
    pixel_mean: List[float] = LIST_FIELD(
        arrList=[0.485, 0.456, 0.406],
        description="The pixel mean for image normalization.",
        display_name="pixel mean"
    )
    pixel_std: List[float] = LIST_FIELD(
        arrList=[0.226, 0.226, 0.226],
        description="The pixel standard deviation for image normalization.",
        display_name="pixel std"
    )
    prob: float = FLOAT_FIELD(
        value=0.5,
        math_cond="> 0.0",
        display_name="random horizontal flipping probability",
        description="The random horizontal flipping probability for image augmentation",
    )
    re_prob: float = FLOAT_FIELD(
        value=0.5,
        math_cond="> 0.0",
        display_name="random erasing probability",
        description="The random erasing probability for image augmentation",
    )
    gaussian_blur: GaussianBlur = DATACLASS_FIELD(
        GaussianBlur(),
        description="The Gaussian blur configuration for the model.",
        display_name="gaussian_blur"
    )
    color_augmentation: ColorAugmentation = DATACLASS_FIELD(
        ColorAugmentation(),
        description="The color augmentation configuration for the model.",
        display_name="color augmentation"
    )
    random_rotation: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="If True, random rotations at 0 ~ 180 degrees to the input data are applied",
        display_name="random rotation augmentation"
    )
    num_instance: int = INT_FIELD(
        value=4,
        default_value=4,
        description="The number of image instances of the same object in a batch",
        display_name="num_instance",
    )
