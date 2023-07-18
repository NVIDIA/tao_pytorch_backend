# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MAL/blob/main/LICENSE
"""COCO dataset."""

from nvidia_tao_pytorch.cv.mal.datasets.voc import InstSegVOC, BoxLabelVOC, InstSegVOCwithBoxInput


class BoxLabelCOCO(BoxLabelVOC):
    """Dataset to load COCO box labels."""

    def get_category_mapping(self):
        """Category mapping."""
        categories = self.coco.dataset['categories']
        self.cat_mapping = {cat['id']: idx + 1 for idx, cat in enumerate(categories)}


class InstSegCOCO(InstSegVOC):
    """Dataset to load COCO instance segmentation labels."""

    def get_category_mapping(self):
        """Category mapping."""
        categories = self.coco.dataset['categories']
        self.cat_mapping = {cat['id']: idx + 1 for idx, cat in enumerate(categories)}


class InstSegCOCOwithBoxInput(InstSegVOCwithBoxInput):
    """Dataset to load COCO labels with only box input."""

    def get_category_mapping(self):
        """Category mapping."""
        categories = self.coco.dataset['categories']
        self.cat_mapping = {cat['id']: idx + 1 for idx, cat in enumerate(categories)}
