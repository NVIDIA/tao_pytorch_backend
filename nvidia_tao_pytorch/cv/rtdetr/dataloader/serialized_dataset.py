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

"""
List serialization code adopted from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py
"""
import torch
from torchvision import tv_tensors

from typing import List, Any
from PIL import Image

from nvidia_tao_pytorch.cv.deformable_detr.utils.data_source_config import build_data_source_lists
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.serialized_dataset import load_coco_json, SerializedDatasetFromList
from nvidia_tao_pytorch.cv.rtdetr.dataloader.od_dataset import mscoco_category2label
from nvidia_tao_pytorch.core.distributed.comm import get_local_rank


def build_shm_dataset(data_sources, transforms, remap_mscoco_category=False):
    """Preload the COCO ann lists to prevent memory leakage from Python.

    Args:
        data_sources (str): list of different data sources.
        transforms (dict): augmentations to apply.
    """
    # grab all the json files and concate them into one single dataset
    data_source_list = build_data_source_lists(data_sources)
    dataset_list = []
    for data_source in data_source_list:
        image_dir = data_source.image_dir
        for _json_file in data_source.dataset_files:
            if get_local_rank() == 0:
                dl = load_coco_json(_json_file, image_root=image_dir)
                dataset_list.extend(dl)
    dataset = RTSerializedDatasetFromList(dataset_list,
                                          transforms=transforms,
                                          remap_mscoco_category=remap_mscoco_category)
    return dataset


class RTSerializedDatasetFromList(SerializedDatasetFromList):
    """
    Hold memory using serialized objects so that data loader workers can use
    shared RAM from master process instead of making a copy in each subprocess.
    """

    def __init__(self, lst, transforms=None, remap_mscoco_category=False):
        """Initialize the Serialized Shared Memory COCO-based Dataset.

        Reference from this blog: https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/

        Args:
            lst (list): list of dataset dicts.
            transforms (dict): augmentations to apply.
        """
        super(RTSerializedDatasetFromList, self).__init__(lst, transforms)
        self.remap_mscoco_category = remap_mscoco_category

    def _process_image_target(self, image: Image.Image, target: List[Any], img_id: int):
        """Process the image and target given image id.

        Args:
            image (PIL.Image): Loaded Pillow Image .
            target (list): Loaded annotation given img_id.
            img_id (int): image id to load.

        Returns:
            target (dict): pre-processed target.
        """
        width, height = image.size
        image_id = torch.tensor([img_id])

        boxes = [obj["bbox"] for obj in target]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        # RT-DETR original src code remap coco category from 1 to 81
        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in target]
        else:
            classes = [obj["category_id"] for obj in target]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format='XYXY',
            canvas_size=image.size[::-1]  # h w
        )

        classes = classes[keep]

        area = torch.tensor([obj["area"] for obj in target])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in target])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(height), int(width)])
        target["size"] = torch.as_tensor([int(height), int(width)])

        return target
