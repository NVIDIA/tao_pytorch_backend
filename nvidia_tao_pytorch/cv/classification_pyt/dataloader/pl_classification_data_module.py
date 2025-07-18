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

"""classification Data Module"""

from typing import Optional
from torch.utils.data import DataLoader, distributed, RandomSampler, BatchSampler
import pytorch_lightning as pl

from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.dataset import CLDataset


class CLDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Classification
    """

    def __init__(self, dataset_config):
        """
        Lightning DataModule Initialization

        Args:
            dataset_config (dict): Configuration for the dataset
        """
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = dataset_config["batch_size"]
        self.num_workers = dataset_config["workers"]
        self.root_dir = dataset_config["root_dir"]
        self.img_size = dataset_config["img_size"]
        self.dataset = dataset_config["dataset"]
        self.augmentation = dataset_config["augmentation"]

    def setup(self, stage: Optional[str] = None):
        """
        Setup the dataset

        Args:
            stage (str): Stage of the dataset
        """
        is_distributed = is_dist_avail_and_initialized()

        if stage == "fit" or stage is None:
            if self.dataset == "CLDataset":
                self.train_dataset = CLDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split="train",
                    img_size=self.img_size,
                    to_tensor=True,
                    data_path=self.dataset_config["train_dataset"]["images_dir"],
                    nolabel_folder=self.dataset_config["train_nolabel"]["folder_path"],
                )
                self.val_dataset = CLDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split="val",
                    img_size=self.img_size,
                    to_tensor=True,
                    data_path=self.dataset_config["val_dataset"]["images_dir"],
                )
            else:
                raise NotImplementedError(
                    "Wrong dataset name %s (choose one from [CLDataset,])"
                    % self.dataset
                )
            if is_distributed:
                self.train_sampler = distributed.DistributedSampler(
                    self.train_dataset, shuffle=True
                )
            else:
                self.train_sampler = RandomSampler(self.train_dataset)

        if stage == "test" or stage is None:
            if self.dataset == "CLDataset":
                self.test_dataset = CLDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split="val",
                    img_size=self.img_size,
                    to_tensor=True,
                    data_path=self.dataset_config["val_dataset"]["images_dir"],
                )
            else:
                raise NotImplementedError(
                    "Wrong dataset name %s (choose one from [CLDataset,])"
                    % self.dataset
                )

        if stage == "predict" or stage is None:
            if self.dataset == "CLDataset":
                self.predict_dataset = CLDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split="test",
                    img_size=self.img_size,
                    to_tensor=True,
                    data_path=self.dataset_config["test_dataset"]["images_dir"],
                )
            else:
                raise NotImplementedError(
                    "Wrong dataset name %s (choose one from [CLDataset,])"
                    % self.dataset
                )

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_loader = DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_sampler=BatchSampler(
                self.train_sampler, self.batch_size, drop_last=False
            ),
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
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
            shuffle=False,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
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
            pin_memory=False,
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
            pin_memory=False,
        )
        return predict_loader
