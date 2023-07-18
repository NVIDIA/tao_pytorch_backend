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

"""Object Detection dataset."""

import torch
from typing import Optional
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.transforms import build_transforms
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import collate_fn, is_dist_avail_and_initialized
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.od_dataset import ODPredictDataset, ODDataset
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.serialized_dataset import build_shm_dataset
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.sampler import UniformSampler, NonUniformSampler, DefaultSampler
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import get_world_size, get_global_rank


class ODDataModule(pl.LightningDataModule):
    """Lightning DataModule for Object Detection."""

    def __init__(self, dataset_config):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration

        """
        super().__init__()
        self.dataset_config = dataset_config
        self.augmentation_config = dataset_config["augmentation"]
        self.batch_size = dataset_config["batch_size"]
        self.num_workers = dataset_config["workers"]
        self.num_classes = dataset_config["num_classes"]
        self.pin_memory = dataset_config["pin_memory"]

    def setup(self, stage: Optional[str] = None):
        """ Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test).

        Args:
            stage (str): stage options from fit, test, predict or None.

        """
        is_distributed = is_dist_avail_and_initialized()

        if stage in ('fit', None):
            # check pytorch distributed is set or not
            train_data_sources = self.dataset_config["train_data_sources"]
            train_transform = build_transforms(self.augmentation_config, dataset_mode='train')
            val_data_sources = self.dataset_config["val_data_sources"]
            val_transform = build_transforms(self.augmentation_config, dataset_mode='val')
            if self.dataset_config["dataset_type"] == "serialized":
                self.train_dataset = build_shm_dataset(train_data_sources, train_transform)
                if is_distributed:
                    self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
                else:
                    self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset)

                # prep validation
                self.val_dataset = build_shm_dataset(val_data_sources, val_transform)
                if is_distributed:
                    self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
                else:
                    self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
            else:
                data_sampler = self.dataset_config["train_sampler"]
                self.val_dataset = DefaultSampler(val_data_sources, is_distributed, transforms=val_transform).build_data_source()
                if is_distributed:  # distributed training
                    if data_sampler == "default_sampler":
                        self.train_dataset, self.train_sampler = DefaultSampler(train_data_sources, is_distributed, transforms=train_transform).get_sampler()
                        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
                    elif data_sampler == "non_uniform_sampler":
                        # manual partial data loading for each GPU. Use this for large dataset which can't fit into the memory, sampler is Default sampler
                        global_rank = get_global_rank()
                        num_gpus = get_world_size()
                        self.train_dataset, self.train_sampler = NonUniformSampler(train_data_sources, transforms=train_transform).get_sampler(global_rank, num_gpus)
                        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
                    elif data_sampler == "uniform_sampler":
                        # manual partial data loading for each GPU. Use this for large dataset which can't fit into the memory, sampler is Uniform Distribution Sampler
                        global_rank = get_global_rank()
                        num_gpus = get_world_size()
                        self.train_dataset, self.train_sampler = UniformSampler(train_data_sources, transforms=train_transform).get_sampler(global_rank, num_gpus)
                        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
                    else:
                        raise NotImplementedError("Sampler {} is not implemented. Use DefaultSampler or UniformSampler".format(data_sampler))
                else:  # Non-distributed learning
                    if data_sampler == "default_sampler":
                        self.train_dataset, self.train_sampler = DefaultSampler(train_data_sources, is_distributed, transforms=train_transform).get_sampler()
                        self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
                    else:
                        raise NotImplementedError("Sampler {} is not implemented for this type of input data. Use DefaultSampler in data_sampler".format(data_sampler))

        # Assign test dataset for use in dataloader
        if stage in ('test', None):
            test_data_sources = self.dataset_config["test_data_sources"]
            self.test_root = test_data_sources.get("image_dir", "")
            test_json = test_data_sources.get("json_file", "")

            test_transforms = build_transforms(self.augmentation_config, dataset_mode='eval')
            if self.dataset_config["dataset_type"] == "serialized":
                self.test_dataset = build_shm_dataset(test_data_sources, transforms=test_transforms)
            else:
                self.test_dataset = ODDataset(dataset_dir=self.test_root, json_file=test_json, transforms=test_transforms)

        # Assign predict dataset for use in dataloader
        if stage in ('predict', None):
            pred_data_sources = self.dataset_config["infer_data_sources"]
            pred_list = pred_data_sources.get("image_dir", [])
            if isinstance(pred_list, str):
                pred_list = [pred_list]
            classmap = pred_data_sources.get("classmap", "")
            self.pred_dataset = ODPredictDataset(pred_list, classmap, transforms=build_transforms(self.augmentation_config, dataset_mode='infer'))

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_loader = DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            batch_sampler=torch.utils.data.BatchSampler(self.train_sampler, self.batch_size, drop_last=True)
        )
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            PyTorch DataLoader used for validation.
        """
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
            sampler=self.val_sampler)

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            PyTorch DataLoader used for evaluation.
        """
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn)

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            PyTorch DataLoader used for inference.
        """
        return DataLoader(
            self.pred_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn)
