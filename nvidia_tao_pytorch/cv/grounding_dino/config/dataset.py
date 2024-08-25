# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    DICT_FIELD,
    STR_FIELD,
)


@dataclass
class GDINOAugmentationConfig:
    """Augmentation config."""

    scales: List[int] = LIST_FIELD(
        arrList=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        description="A list of sizes to perform random resize.",
        display_name="scales"
    )
    input_mean: List[float] = LIST_FIELD(
        arrList=[0.485, 0.456, 0.406],
        description="The input mean for RGB frames",
        display_name="input mean per pixel"
    )
    input_std: List[float] = LIST_FIELD(
        arrList=[0.229, 0.224, 0.225],
        description="The input standard deviation per pixel for RGB frames",
        display_name="input std per pixel"
    )
    train_random_resize: List[int] = LIST_FIELD(
        arrList=[400, 500, 600],
        description="A list of sizes to perform random resize for training data",
        display_name="train random resize dimensions"
    )
    horizontal_flip_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="The probability for horizonal flip during training",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="horizontal flip probability"
    )
    train_random_crop_min: int = INT_FIELD(
        value=384,
        valid_min=1,
        valid_max="inf",
        description="The minimum random crop size for training data",
        automl_enabled="TRUE",
        display_name="minumum random crop size"
    )
    train_random_crop_max: int = INT_FIELD(
        value=600,
        valid_min=1,
        valid_max="inf",
        description="The maximum random crop size for training data",
        automl_enabled="TRUE",
        display_name="Maximum random crop size"
    )
    random_resize_max_size: int = INT_FIELD(
        value=1333,
        valid_min=1,
        valid_max="inf",
        description="The maximum random resize size for training data",
        automl_enabled="TRUE",
        display_name="random resize max size"
    )
    test_random_resize: int = INT_FIELD(
        value=800,
        valid_min=1,
        valid_max="inf",
        description="The random resize size for test data",
        automl_enabled="TRUE",
        display_name="random resize max size"
    )
    fixed_padding: bool = BOOL_FIELD(
        value="TRUE",
        default_value="TRUE",
        description="""A flag specifying whether to resize the image (with no padding) to
                     (sorted(scales[-1]), random_resize_max_size) to prevent a CPU " \
                    memory leak. """,
        display_name="fixed padding"
    )
    fixed_random_crop: Optional[int] = INT_FIELD(
        value=None,
        default_value=1024,
        description="""A flag to enable Large Scale Jittering, which is used for ViT backbones.
                    The resulting image resolution is fixed to fixed_random_crop.""",
        display_name="fixed random crop",
        valid_min=1,
        valid_max="inf"
    )


@dataclass
class GDINODatasetConfig:
    """Dataset config."""

    train_data_sources: Optional[List[Dict[str, str]]] = LIST_FIELD(
        arrList=None,
        default_value=[{"image_dir": "", "json_file": "", "label_map": ""}, {"image_dir": "", "json_file": ""}],
        description="""The list of data sources for training:
                    * image_dir : The directory that contains the training images
                    * json_file : The path of the JSONL file, which uses training-annotation ODVG format
                    * label_map: (Optional) The path of the label mapping only required for detection dataset""",
        display_name="train data sources",
    )
    val_data_sources: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        arrList=None,
        default_value={"image_dir": "", "json_file": ""},
        description="""The data source for validation:
                    * image_dir : The directory that contains the validation images
                    * json_file : The path of the JSON file, which uses validation-annotation COCO format.
                    Note that category id needs to start from 0 if we want to calculate validation loss.
                    Run Data Services annotation convert to making the categories contiguous.""",
        display_name="validation data sources",
    )
    test_data_sources: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        arrList=None,
        default_value={"image_dir": "", "json_file": ""},
        description="""The data source for testing:
                    * image_dir : The directory that contains the test images
                    * json_file : The path of the JSON file, which uses test-annotation COCO format""",
        display_name="test data sources",
    )
    infer_data_sources: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        arrList=None,
        default_value={"image_dir": [""], "captions": [""]},
        description="""The data source for inference:
                    * image_dir : The list of directories that contains the inference images
                    * captions : The list of caption to run inference""",
        display_name="infer data sources",
    )
    batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        valid_max="inf",
        description="The batch size for training and validation",
        automl_enabled="TRUE",
        display_name="batch size"
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="The number of parallel workers processing data",
        automl_enabled="TRUE",
        display_name="batch size"
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="pin_memory",
        description="""Flag to enable the dataloader to allocated pagelocked memory for faster
                    of data between the CPU and GPU."""
    )
    dataset_type: str = STR_FIELD(
        value="serialized",
        default_value="serialized",
        display_name="dataset type",
        description="""If set to default, we follow the standard map-style dataset structure
                    from torch which loads ODVG annotation in every subprocess. This leads to redudant
                    copy of data and can cause RAM to explod if `workers` is high. If set to serialized,
                    the data is serialized through pickle and `torch.Tensor` that allows the data to be shared
                    across subprocess. As a result, RAM usage can be greatly improved.""",
        valid_options=",".join(["serialized", "default"])
    )
    max_labels: int = INT_FIELD(
        value=50,
        default_value=50,
        description="""The total number of labels to sample from. After sampling positive labels,
                    we randomly sample negative samples so that total number of labels equal to `max_labels`.
                    For detection dataset, negative labels are categories not present in the image.
                    For grounding dataset, negative labels are phrases in the original caption not present in the image.
                    Setting higher `max_labels` may improve robustness of the model with the cost of longer training time.""",
        math_cond=">0",
        valid_min=1,
        valid_max="inf",
        display_name="max labels"
    )
    eval_class_ids: Optional[List[int]] = LIST_FIELD(
        arrList=None,
        default_value=[1],
        description="""IDs of the classes for evaluation.""",
        display_name="eval class ids",
    )
    augmentation: GDINOAugmentationConfig = DATACLASS_FIELD(
        GDINOAugmentationConfig(),
        description="Configuration parameters for data augmentation",
        display_name="augmentation",
    )
