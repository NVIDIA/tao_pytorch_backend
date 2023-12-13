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

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetMapper

from nvidia_tao_pytorch.cv.odise.modeling.wrapper.pano_wrapper import OpenPanopticInference
from nvidia_tao_pytorch.cv.odise.data import build_d2_test_dataloader, get_openseg_labels

from nvidia_tao_pytorch.cv.odise.evaluation.d2_evaluator import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    SemSegEvaluator,
    InstanceSegEvaluator,
)

coco133_open_eval = OmegaConf.create()
coco133_open_eval.loader = L(build_d2_test_dataloader)(
    dataset=L(get_detection_dataset_dicts)(
        names="coco_2017_val_panoptic_with_sem_seg", filter_empty=False
    ),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=1024,
                max_size=2560,
                sample_style="choice",
            ),
        ],
        image_format="RGB",
    ),
    local_batch_size=1,
    num_workers=1,
)

coco133_open_eval.wrapper = L(OpenPanopticInference)(
    labels=L(get_openseg_labels)(dataset="coco_panoptic", prompt_engineered=True),
    metadata=L(MetadataCatalog.get)(name=coco133_open_eval.loader.dataset.names),
)

coco133_open_eval.evaluator = [
    L(COCOEvaluator)(
        dataset_name="${...loader.dataset.names}",
        tasks=("segm",),
    ),
    L(SemSegEvaluator)(
        dataset_name="${...loader.dataset.names}",
    ),
    L(COCOPanopticEvaluator)(
        dataset_name="${...loader.dataset.names}",
    ),
]

ade150_open_eval = OmegaConf.create()
ade150_open_eval.loader = L(build_d2_test_dataloader)(
    dataset=L(get_detection_dataset_dicts)(names="ade20k_panoptic_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=1024,
                max_size=2560,
                sample_style="choice",
            ),
        ],
        image_format="RGB",
    ),
    local_batch_size=1,
    num_workers=1,
)

ade150_open_eval.wrapper = L(OpenPanopticInference)(
    labels=L(get_openseg_labels)(dataset="ade20k_150", prompt_engineered=True),
    metadata=L(MetadataCatalog.get)(name=ade150_open_eval.loader.dataset.names),
)

ade150_open_eval.evaluator = [
    L(InstanceSegEvaluator)(
        dataset_name="${...loader.dataset.names}",
        tasks=("segm",),
    ),
    L(SemSegEvaluator)(
        dataset_name="${...loader.dataset.names}",
    ),
    L(COCOPanopticEvaluator)(
        dataset_name="${...loader.dataset.names}",
    ),
]

ade847_open_eval = OmegaConf.create()
ade847_open_eval.loader = L(build_d2_test_dataloader)(
    dataset=L(get_detection_dataset_dicts)(names="ade20k_full_sem_seg_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=1024,
                max_size=2560,
                sample_style="choice",
            ),
        ],
        image_format="RGB",
    ),
    local_batch_size=1,
    num_workers=1,
)

ade847_open_eval.wrapper = L(OpenPanopticInference)(
    labels=L(get_openseg_labels)(dataset="ade20k_847", prompt_engineered=True),
    metadata=L(MetadataCatalog.get)(name=ade847_open_eval.loader.dataset.names),
    semantic_on=True,
    instance_on=False,
    panoptic_on=False,
)

ade847_open_eval.evaluator = [
    L(SemSegEvaluator)(
        dataset_name="${...loader.dataset.names}",
    ),
]

ctx59_open_eval = OmegaConf.create()
ctx59_open_eval.loader = L(build_d2_test_dataloader)(
    dataset=L(get_detection_dataset_dicts)(names="ctx59_sem_seg_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=1024,
                max_size=2560,
                sample_style="choice",
            ),
        ],
        image_format="RGB",
    ),
    local_batch_size=1,
    num_workers=1,
)

ctx59_open_eval.wrapper = L(OpenPanopticInference)(
    labels=L(get_openseg_labels)(dataset="pascal_context_59", prompt_engineered=True),
    metadata=L(MetadataCatalog.get)(name=ctx59_open_eval.loader.dataset.names),
    semantic_on=True,
    instance_on=False,
    panoptic_on=False,
)

ctx59_open_eval.evaluator = [
    L(SemSegEvaluator)(
        dataset_name="${...loader.dataset.names}",
    ),
]

ctx459_open_eval = OmegaConf.create()
ctx459_open_eval.loader = L(build_d2_test_dataloader)(
    dataset=L(get_detection_dataset_dicts)(names="ctx459_sem_seg_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=1024,
                max_size=2560,
                sample_style="choice",
            ),
        ],
        image_format="RGB",
    ),
    local_batch_size=1,
    num_workers=1,
)

ctx459_open_eval.wrapper = L(OpenPanopticInference)(
    labels=L(get_openseg_labels)(dataset="pascal_context_459", prompt_engineered=True),
    metadata=L(MetadataCatalog.get)(name=ctx459_open_eval.loader.dataset.names),
    semantic_on=True,
    instance_on=False,
    panoptic_on=False,
)

ctx459_open_eval.evaluator = [
    L(SemSegEvaluator)(
        dataset_name="${...loader.dataset.names}",
    ),
]

pas21_open_eval = OmegaConf.create()
pas21_open_eval.loader = L(build_d2_test_dataloader)(
    dataset=L(get_detection_dataset_dicts)(names="pascal21_sem_seg_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=1024,
                max_size=2560,
                sample_style="choice",
            ),
        ],
        image_format="RGB",
    ),
    local_batch_size=1,
    num_workers=1,
)

pas21_open_eval.wrapper = L(OpenPanopticInference)(
    labels=L(get_openseg_labels)(dataset="pascal_voc_21", prompt_engineered=True),
    metadata=L(MetadataCatalog.get)(name=pas21_open_eval.loader.dataset.names),
    semantic_on=True,
    instance_on=False,
    panoptic_on=False,
)

pas21_open_eval.evaluator = [
    L(SemSegEvaluator)(
        dataset_name="${...loader.dataset.names}",
    ),
]
