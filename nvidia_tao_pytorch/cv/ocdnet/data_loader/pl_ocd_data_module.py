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

"""Object Detection dataset."""

from typing import Optional
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from nvidia_tao_pytorch.cv.ocdnet.data_loader.build_dataloader import get_dataloader
from nvidia_tao_pytorch.cv.ocdnet.data_loader.dataset import CustomImageDataset


class OCDDataModule(pl.LightningDataModule):
    """Lightning DataModule for OCDNet."""

    def __init__(self, experiment_spec):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration

        """
        super().__init__()
        self.experiment_spec = experiment_spec
        self.dataset_config = experiment_spec['dataset']
        self.model_config = experiment_spec['model']

        self.train_dataset_config = self.dataset_config["train_dataset"]
        self.validate_dataset_config = self.dataset_config["validate_dataset"]

    def setup(self, stage: Optional[str] = None):
        """ Prepares for each dataloader

        Args:
            stage (str): stage options from fit, validate, test, predict or None.

        """
        if stage == 'fit':
            self.train_loader = get_dataloader(self.train_dataset_config, self.experiment_spec['train']['num_gpus'] > 1)
            assert self.train_loader is not None, "Train loader does not exist."
            self.train_loader_len = len(self.train_loader)
        elif stage == 'predict':
            input_path = self.experiment_spec['inference']["input_folder"]
            width = self.experiment_spec['inference']['width']
            height = self.experiment_spec['inference']['height']
            img_mode = self.experiment_spec['inference']['img_mode']
            self.predict_dataset = CustomImageDataset(input_path, width, height, img_mode)

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader (Dataloader): Traininig Data.

        """
        return self.train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader (Dataloader): Validation Data.

        """
        if 'validate_dataset' in self.dataset_config:
            val_loader = get_dataloader(self.validate_dataset_config, False)
        else:
            val_loader = None

        return val_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader (Dataloader): Evaluation Data.

        """
        test_loader = get_dataloader(self.validate_dataset_config, False)

        return test_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader (Dataloader): Inference Data.

        """
        predict_loader = DataLoader(self.predict_dataset, batch_size=1)

        return predict_loader
