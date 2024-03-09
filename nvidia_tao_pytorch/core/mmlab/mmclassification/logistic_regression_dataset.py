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

"""Dataloder Class for Classification - LR Head."""

import os
from torch.utils.data import Dataset


class LRDataset(Dataset):
    """Dataset class for logistic regression.

    Args:
        data_prefix (str): Path prefix for dataset.
        classes (List[str], optional): List of class labels. Defaults to None.
        ann_file (str, optional): Path to annotation file. Defaults to None.
    """

    def __init__(self, data_prefix, classes=None, ann_file=None):
        """Init module"""
        self.data_prefix = data_prefix
        self.class_to_idx = self._load_classes(classes)
        self.img_labels = self._load_annotations(ann_file)

    def _load_classes(self, classes_file):
        if classes_file:
            with open(classes_file, 'r') as file:
                classes = [line.strip() for line in file.readlines()]
            classes.sort()  # Sort classes alphabetically
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            return class_to_idx
        return None

    def _load_annotations(self, ann_file):
        img_labels = []
        if ann_file:
            with open(ann_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    img_path, label = line.strip().split()
                    label = int(label)
                    img_path = os.path.join(self.data_prefix, img_path)
                    img_labels.append((img_path, label))
        else:
            classes = [d for d in os.listdir(self.data_prefix) if os.path.isdir(os.path.join(self.data_prefix, d))]
            if self.class_to_idx is None:
                classes.sort()
                self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            for class_name in classes:
                class_path = os.path.join(self.data_prefix, class_name)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    label = self.class_to_idx[class_name]
                    img_labels.append((img_path, label))
        return img_labels

    def __len__(self):
        """Return number of samples in the dataset"""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """get item"""
        img_path, label = self.img_labels[idx]
        return img_path, label
