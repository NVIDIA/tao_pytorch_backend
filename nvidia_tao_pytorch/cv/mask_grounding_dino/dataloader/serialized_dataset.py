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

"""Serialized dataset to use shared memory to prevent memory leak."""

import os
import torch
import json
from typing import Dict, List
import random
from PIL import Image, ImageOps
import pycocotools.mask as mask_util
import numpy as np

from nvidia_tao_pytorch.core.distributed.comm import get_local_rank
from nvidia_tao_pytorch.core.distributed.serialized_object import TorchShmSerializedList


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


def load_coco_jsonl(jsonl_file: str, image_root: str, labelmap_file: str = None) -> List[Dict]:
    """Load ODVG jsonl file and return list of dictionaries.

    Args:
        jsonl_file (str): Path to the JSONL annotation file.
        image_root (str): Path to root directory of images from the annotations.
        labelmap_file (str): Path to category mapping. Only required for detection task.

    Returns:
        List of meta data.
    """
    with open(jsonl_file, "r") as f:
        metas = [json.loads(line) for line in f]

    if labelmap_file:
        with open(labelmap_file, "r") as f:
            label_map = json.load(f)

    dataset_dicts = []
    for meta in metas:
        meta["file_name"] = os.path.join(image_root, meta["file_name"])
        # optionally add label map info
        if labelmap_file:
            meta["detection"]["label_map"] = label_map
        dataset_dicts.append(meta)
    return dataset_dicts


def build_shm_dataset(data_sources, transforms, max_labels=80):
    """Preload the COCO ann lists to prevent memory leakage from Python.

    Args:
        data_sources (str): list of different data sources.
        transforms (dict): augmentations to apply.
    """
    # grab all the json files and concate them into one single dataset
    # data_source_list = build_data_source_lists(data_sources)
    dataset_list = []
    for data_source in data_sources:
        label_map = None
        if "label_map" in data_source:
            label_map = data_source.label_map
        if get_local_rank() == 0:
            dl = load_coco_jsonl(data_source.json_file,
                                 image_root=data_source.image_dir,
                                 labelmap_file=label_map)
            dataset_list.extend(dl)
    dataset = ODVGSerializedDatasetFromList(dataset_list,
                                            transforms=transforms,
                                            max_labels=max_labels)
    return dataset


class ODVGSerializedDatasetFromList(torch.utils.data.Dataset):
    """
    Hold memory using serialized objects so that data loader workers can use
    shared RAM from master process instead of making a copy in each subprocess.
    """

    def __init__(self, lst, transforms=None, max_labels=80, has_mask=True):
        """Initialize the Serialized Shared Memory ODVG-based Dataset.

        Reference from this blog: https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/

        Args:
            lst (list): list of dataset dicts.
            transforms (dict): augmentations to apply.
            max_labels (int): number of pos labels + sampled neg labels
        """
        self.transforms = transforms
        self.max_labels = max_labels
        self.has_mask = has_mask
        self.metas = TorchShmSerializedList(lst)

    def __len__(self):
        """__len__"""
        return len(self.metas)

    def __getitem__(self, index: int):
        """Get image, target, image_path given index,

        Args:
            index (int): index of the image id to load.

        Returns:
            (image, target): pre-processed image and target for the model.
        """
        meta = self.metas[index]
        image_path = meta['file_name']
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")

        w, h = image.size

        if "detection" in meta:
            dataset_mode = "OD"
        elif "grounding" in meta:
            dataset_mode = "VG"
        else:
            raise NotImplementedError("Missing either 'detection' / 'grounding' key in the annotation")

        if dataset_mode == "OD":
            anno = meta["detection"]
            label_map = anno["label_map"]
            label_index = set(label_map.keys())
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
            neg_labels = label_index.difference(pos_labels)

            vg_labels = list(pos_labels)
            num_to_add = min(len(neg_labels), self.max_labels - len(pos_labels))
            if num_to_add > 0:
                vg_labels.extend(random.sample(neg_labels, num_to_add))

            # shuffle
            for i in range(len(vg_labels) - 1, 0, -1):
                j = random.randint(0, i)
                vg_labels[i], vg_labels[j] = vg_labels[j], vg_labels[i]

            caption_list = [label_map[lb] for lb in vg_labels]
            caption_dict = {item: index for index, item in enumerate(caption_list)}

            caption = ' . '.join(caption_list) + ' .'
            classes = [caption_dict[label_map[str(obj["label"])]] for obj in instances]

            boxes, classes, segms = self.preprocess_boxes(boxes, classes, segms, w, h)

        elif dataset_mode == "VG":
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
