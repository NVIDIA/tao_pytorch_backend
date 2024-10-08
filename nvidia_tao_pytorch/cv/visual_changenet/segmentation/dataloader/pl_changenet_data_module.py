# Copyright (c) 2023 Chaminda Bandara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Original source taken from https://github.com/wgcban/ChangeFormer
#
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

"""Visual ChangeNet Data Module"""

from typing import Optional
from torch.utils.data import DataLoader, distributed, RandomSampler, BatchSampler
import pytorch_lightning as pl

from nvidia_tao_pytorch.cv.visual_changenet.segmentation.dataloader.cn_dataset import CNDataset
from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized


class CNDataModule(pl.LightningDataModule):
    """Lightning DataModule for Object Detection."""

    def __init__(self, dataset_config):
        """ Lightning DataModule Initialization

        Args:
            dataset_config: dataset configuration

        """
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = dataset_config["batch_size"]
        self.num_workers = dataset_config["workers"]
        self.root_dir = dataset_config["root_dir"]
        self.label_transform = dataset_config["label_transform"]
        self.img_size = dataset_config["img_size"]
        self.dataset = dataset_config["dataset"]
        self.image_folder_name = dataset_config["image_folder_name"]
        self.change_image_folder_name = dataset_config["change_image_folder_name"]
        self.list_folder_name = dataset_config["list_folder_name"]
        self.annotation_folder_name = dataset_config["annotation_folder_name"]
        self.augmentation = dataset_config["augmentation"]
        self.label_suffix = dataset_config["label_suffix"]

    def setup(self, stage: Optional[str] = None):
        """ Prepares for each dataloader

        Args:
            stage (str): stage options from fit, validate, test, predict or None.

        """
        is_distributed = is_dist_avail_and_initialized()

        if stage in ('fit', None):

            split = self.dataset_config['train_split']
            split_val = self.dataset_config['validation_split']

            if self.dataset == 'CNDataset':
                self.train_dataset = CNDataset(root_dir=self.root_dir, split=split,
                                               img_size=self.img_size, is_train=True,
                                               label_transform=self.label_transform, a_dir=self.image_folder_name,
                                               b_dir=self.change_image_folder_name, label_dir=self.annotation_folder_name,
                                               list_dir=self.list_folder_name, augmentation=self.augmentation, label_suffix=self.label_suffix)
                self.val_dataset = CNDataset(root_dir=self.root_dir, split=split_val,
                                             img_size=self.img_size, is_train=False,
                                             label_transform=self.label_transform, a_dir=self.image_folder_name,
                                             b_dir=self.change_image_folder_name, label_dir=self.annotation_folder_name,
                                             list_dir=self.list_folder_name, label_suffix=self.label_suffix)
            else:
                raise NotImplementedError(
                    'Wrong dataset name %s (choose one from [CNDataset,])'
                    % self.dataset)
            if is_distributed:
                self.train_sampler = distributed.DistributedSampler(self.train_dataset, shuffle=True)
            else:
                self.train_sampler = RandomSampler(self.train_dataset)

        # Assign test dataset for use in dataloader
        if stage in ('test', None):
            split = self.dataset_config['test_split']

            if self.dataset == 'CNDataset':
                self.test_dataset = CNDataset(root_dir=self.root_dir, split=split,
                                              img_size=self.img_size, is_train=False,
                                              label_transform=self.label_transform, a_dir=self.image_folder_name,
                                              b_dir=self.change_image_folder_name, label_dir=self.annotation_folder_name,
                                              list_dir=self.list_folder_name, label_suffix=self.label_suffix)
            else:  # TODO: check if this needed
                raise NotImplementedError(
                    'Wrong dataset name %s (choose one from [CNDataset])'
                    % self.dataset)

        if stage in ('predict', None):
            split = self.dataset_config['predict_split']

            if self.dataset == 'CNDataset':
                self.predict_dataset = CNDataset(root_dir=self.root_dir, split=split,
                                                 img_size=self.img_size, is_train=False,
                                                 label_transform=self.label_transform, a_dir=self.image_folder_name,
                                                 b_dir=self.change_image_folder_name, label_dir=self.annotation_folder_name,
                                                 list_dir=self.list_folder_name, label_suffix=self.label_suffix)
            else:
                raise NotImplementedError(
                    'Wrong dataset name %s (choose one from [CNDataset])'
                    % self.dataset)

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_loader = DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_sampler=BatchSampler(self.train_sampler, self.batch_size, drop_last=True)
        )
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        val_loader = DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True
        )
        return val_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader: PyTorch DataLoader used for evaluation.
        """
        test_loader = DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )
        return test_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        predict_loader = DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )
        return predict_loader
