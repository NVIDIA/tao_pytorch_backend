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

"""Object Detection Visual Genome Dataset."""

import json
import random
import os
from typing import Callable, Optional, List

from PIL import Image
import numpy as np
import pycocotools.mask as mask_util

import torch
from torchvision.datasets.vision import VisionDataset


def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)


class ODVGDataset(VisionDataset):
    """Object Detection Visual Genome Dataset."""

    def __init__(
        self,
        root: str,
        anno: str,
        label_map_anno: str = None,
        max_labels: int = 80,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        has_mask: bool = False,
    ) -> None:
        """Initialize ODVG dataset.
        Args:
            root (string): Root directory where images are downloaded to.
            anno (string): Path to json annotation file.
            label_map_anno (string):  Path to json label mapping file. Only for Object Detection
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.PILToTensor``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            transforms (callable, optional): A function/transform that takes input sample and its target as entry
                and returns a transformed version.
        """
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.dataset_mode = "OD" if label_map_anno else "VG"
        self.max_labels = max_labels
        self.has_mask = has_mask
        if self.dataset_mode == "OD":
            self.load_label_map(label_map_anno)
        self._load_metas(anno)
        self.get_dataset_info()

    def load_label_map(self, label_map_anno):
        """Load the label map json file for detection dataset."""
        with open(label_map_anno, 'r') as file:
            self.label_map = json.load(file)
        self.label_index = set(self.label_map.keys())

    def _load_metas(self, anno):
        """Load ODVG jsonl file"""
        with open(anno, 'r')as f:
            self.metas = [json.loads(line) for line in f]

    def get_dataset_info(self):
        """print dataset info."""
        print(f"  == total images: {len(self)}")
        if self.dataset_mode == "OD":
            print(f"  == total labels: {len(self.label_map)}")

    def prepare_masks(self, masks, h, w):
        """Preprocess mask."""
        segms = []
        for mask in masks:
            if isinstance(mask, list):
                # polygon
                segms.append(polygons_to_bitmask(mask, h, w))
            elif isinstance(mask, dict):
                # COCO RLE
                if 'counts' in mask.keys():
                    if isinstance(mask['counts'], list):
                        rle = mask_util.frPyObjects(mask, h, w)
                    else:
                        rle = mask
                else:
                    raise ValueError("Wrong mask format")
                segms.append(mask_util.decode(rle))
            elif isinstance(mask, np.ndarray):
                assert mask.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                    mask.ndim
                )
                # mask array
                segms.append(mask)
            else:
                raise ValueError(
                    "Cannot convert segmentation of type '{}' to BitMasks!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict, or a binary segmentation mask "
                    " in a 2D numpy array of shape HxW.".format(type(mask))
                )
        segms = torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in segms])
        return segms

    def __getitem__(self, index: int):
        """return image / target."""
        meta = self.metas[index]
        rel_path = meta["file_name"]
        abs_path = os.path.join(self.root, rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{self.root} {rel_path} {abs_path} not found.")
        image = Image.open(abs_path).convert('RGB')
        w, h = image.size
        if self.dataset_mode == "OD":
            anno = meta["detection"]
            instances = [obj for obj in anno["instances"]]
            boxes = [obj["bbox"] for obj in instances]
            segms = None
            if self.has_mask:
                masks = [obj["mask"] for obj in instances]
                assert len(boxes) == len(masks), "The number of boxes and masks don't match."
                if len(boxes) == 0:
                    segms = torch.zeros((0, h, w))
                else:
                    segms = self.prepare_masks(masks, h, w)

            # generate vg_labels
            # pos bbox labels
            ori_classes = [str(obj["label"]) for obj in instances]
            pos_labels = set(ori_classes)
            # neg bbox labels
            neg_labels = self.label_index.difference(pos_labels)

            vg_labels = list(pos_labels)
            num_to_add = min(len(neg_labels), self.max_labels - len(pos_labels))
            if num_to_add > 0:
                vg_labels.extend(random.sample(neg_labels, num_to_add))

            # shuffle
            for i in range(len(vg_labels) - 1, 0, -1):
                j = random.randint(0, i)
                vg_labels[i], vg_labels[j] = vg_labels[j], vg_labels[i]

            caption_list = [self.label_map[lb] for lb in vg_labels]
            caption_dict = {item: index for index, item in enumerate(caption_list)}

            caption = ' . '.join(caption_list) + ' .'
            classes = [caption_dict[self.label_map[str(obj["label"])]] for obj in instances]

            boxes, classes, segms = self.preprocess_boxes(boxes, classes, segms, w, h)

        elif self.dataset_mode == "VG":
            anno = meta["grounding"]
            instances = [obj for obj in anno["regions"]]
            boxes = [obj["bbox"] for obj in instances]
            segms = None
            if self.has_mask:
                masks = [obj["mask"] for obj in instances]
                assert len(boxes) == len(masks), "The number of boxes and masks don't match."
                if len(boxes) == 0:
                    segms = torch.zeros((0, h, w))
                else:
                    segms = self.prepare_masks(masks, h, w)

            caption_list = [obj["phrase"] for obj in instances]
            c = list(zip(boxes, caption_list))
            random.shuffle(c)
            boxes[:], caption_list[:] = zip(*c)
            uni_caption_list = list(set(caption_list))
            label_map = {}
            for idx in range(len(uni_caption_list)):
                label_map[uni_caption_list[idx]] = idx
            classes = [label_map[cap] for cap in caption_list]
            caption = ' . '.join(uni_caption_list) + ' .'
            caption_list = uni_caption_list

            boxes, classes, segms = self.preprocess_boxes(boxes, classes, segms, w, h)

        target = {}
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["cap_list"] = caption_list
        target["caption"] = caption
        target["boxes"] = boxes
        target["labels"] = classes
        if self.has_mask:
            target["masks"] = segms

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def preprocess_boxes(self, boxes, classes, segms, w, h):
        """Filter boxes and masks."""
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # Clamp the coordinates to the image resolution
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Filter out invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        classes = torch.tensor(classes, dtype=torch.int64)
        classes = classes[keep]
        return boxes, classes, segms[keep] if segms is not None else torch.zeros((0, 1, 1))

    def __len__(self) -> int:
        """return length of the dataset."""
        return len(self.metas)
