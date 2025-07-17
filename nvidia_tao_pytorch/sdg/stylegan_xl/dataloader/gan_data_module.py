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

"""General Data Module"""

import torch
import pytorch_lightning as pl
from nvidia_tao_pytorch.sdg.stylegan_xl.utils import dnnlib
from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized


class GANDataModule(pl.LightningDataModule):
    """Base class for GAN DataModules"""

    def __init__(self, dataset_config):
        """Base initialization for the GANDataModule.

        Args:
            dataset_config: Dataset configuration
        """
        super(GANDataModule, self).__init__()
        self.dataset_config = dataset_config
        self.batch_size = dataset_config['batch_size']
        self.data_loader_kwargs = dnnlib.EasyDict(pin_memory=dataset_config['pin_memory'],
                                                  prefetch_factor=dataset_config['prefetch_factor'],
                                                  num_workers=dataset_config['workers'])

    def _set_unrepeated_samplers(self, dataset):
        """Set up unrepeated samplers for distributed training for evaluation and validation sets.

        Args:
            dataset (Dataset): Dataset for which the sampler is to be set up.

        Returns:
            sampler: A sampler that is either distributed or sequential based on
                     whether distributed training is available.
        """
        is_distributed = is_dist_avail_and_initialized()
        if is_distributed:
            return pl.overrides.distributed.UnrepeatedDistributedSampler(dataset, shuffle=False)
        else:
            return torch.utils.data.SequentialSampler(dataset)

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_dataloader = torch.utils.data.DataLoader(
            shuffle=True,
            dataset=self.training_set,
            batch_size=self.batch_size // self.trainer.world_size,
            **self.data_loader_kwargs
        )
        return train_dataloader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        val_dataloader = torch.utils.data.DataLoader(
            shuffle=False,
            dataset=self.validation_set,
            batch_size=self.batch_size // self.trainer.world_size,
            sampler=self.validation_sampler,
            **self.data_loader_kwargs
        )
        return val_dataloader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader: PyTorch DataLoader used for evaluation.
        """
        evaluation_dataloader = torch.utils.data.DataLoader(
            shuffle=False,
            dataset=self.evaluation_set,
            batch_size=self.batch_size // self.trainer.world_size,
            sampler=self.evaluation_sampler,
            **self.data_loader_kwargs
        )
        return evaluation_dataloader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        predict_loader = torch.utils.data.DataLoader(self.seed_dataset, batch_size=self.batch_size // self.trainer.world_size, shuffle=False)
        return predict_loader
