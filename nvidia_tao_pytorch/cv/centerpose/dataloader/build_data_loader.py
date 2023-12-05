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

"""CenterPose dataset."""

from typing import Optional
from torch.utils.data import DataLoader, distributed, RandomSampler, BatchSampler
import pytorch_lightning as pl

from nvidia_tao_pytorch.cv.centerpose.dataloader.data_feeder import ObjectPoseDataset, CPPredictDataset
from nvidia_tao_pytorch.cv.centerpose.dataloader.augmentation import collate_fn_filtered
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import is_dist_avail_and_initialized


class CPDataModule(pl.LightningDataModule):
    """Lightning DataModule for CenterPose."""

    def __init__(self, dataset_config):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration

        """
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = dataset_config.batch_size
        self.num_workers = dataset_config.workers
        self.pin_memory = dataset_config.pin_memory

    def setup(self, stage: Optional[str] = None):
        """ Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test).

        Args:
            stage (str): stage options from fit, test, predict or None.

        """
        is_distributed = is_dist_avail_and_initialized()

        if stage in ('fit', None):
            # check pytorch distributed is set or not
            self.train_dataset = ObjectPoseDataset(self.dataset_config.train_data, self.dataset_config, 'train')
            self.val_dataset = ObjectPoseDataset(self.dataset_config.val_data, self.dataset_config, 'test')

            if is_distributed:
                self.train_sampler = distributed.DistributedSampler(self.train_dataset, shuffle=True)
            else:
                self.train_sampler = RandomSampler(self.train_dataset)

        # Assign test dataset for use in dataloader
        elif stage in ('test', None):
            self.test_dataset = ObjectPoseDataset(self.dataset_config.test_data, self.dataset_config, 'test')

        # Assign predict dataset for use in dataloader
        elif stage in ('predict', None):
            self.pred_dataset = CPPredictDataset(self.dataset_config)

        else:
            raise NotImplementedError(f"Invalid stage {stage} encountered.")

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_loader = DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn_filtered,
            batch_sampler=BatchSampler(self.train_sampler, self.batch_size, drop_last=True)
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
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn_filtered)

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
            collate_fn=collate_fn_filtered)

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
            collate_fn=collate_fn_filtered)
