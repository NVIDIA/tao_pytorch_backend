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

"""BigDatasetGAN Data Module"""

from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.gan_data_module import GANDataModule
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.bg_dataset import LabelDataset
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.seed_dataset import SeedDataset


class BGDataModule(GANDataModule):
    """Lightning DataModule for BigDatasetGAN."""

    def __init__(self, dataset_config):
        """ Lightning DataModule Initialization

        Args:
            dataset_config: dataset configuration

        """
        super(BGDataModule, self).__init__(dataset_config)
        # assert dataset_config['common']['img_resolution'] == 512, "Current BigDatasetGAN with StyleGAN-XL as backbone only support resolution 512x512 mask generation"
        self.evaluate_data_dir = dataset_config['bigdatasetgan']['test_dataset']['images_dir']
        self.validate_data_dir = dataset_config['bigdatasetgan']['validation_dataset']['images_dir']
        self.train_data_dir = dataset_config['bigdatasetgan']['train_dataset']['images_dir']

    def setup(self, stage=None):
        """ Prepares for each dataloader

        Args:
            stage (str): stage options from fit, validate, test, predict or None.

        """
        if stage in ('fit', None):
            self.training_set = LabelDataset(self.train_data_dir)
            self.validation_set = LabelDataset(self.validate_data_dir)
            self.validation_sampler = self._set_unrepeated_samplers(self.validation_set)

        elif stage == 'test':
            self.evaluation_set = LabelDataset(self.evaluate_data_dir)
            self.evaluation_sampler = self._set_unrepeated_samplers(self.evaluation_set)

        elif stage == 'predict':
            seeds = list(range(self.dataset_config['bigdatasetgan']['infer_dataset']['start_seed'], self.dataset_config['bigdatasetgan']['infer_dataset']['end_seed']))
            self.seed_dataset = SeedDataset(seeds)
            # The default sampler is already UnrepeatedDistributedSampler when "predict"
