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

"""Metric Learning Data Module"""

from typing import Optional
import pytorch_lightning as pl

from nvidia_tao_pytorch.cv.ml_recog.dataloader.build_data_loader import build_dataloader, build_inference_dataloader


class MLDataModule(pl.LightningDataModule):
    """Lightning DataModule for Metric Learning."""

    def __init__(self, experiment_spec):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration

        """
        super().__init__()
        self.experiment_spec = experiment_spec

    def setup(self, stage: Optional[str] = None):
        """ Prepares for each dataloader

        Args:
            stage (str): stage options from fit, validate, test, predict or None.

        """
        if stage == 'fit':
            (self.train_loader, self.query_loader, self.gallery_loader,
                self.dataset_dict) = build_dataloader(cfg=self.experiment_spec, mode="train")
            self.class_dict = self.dataset_dict["query"].class_dict
        elif stage == 'test':
            _, _, self.test_loader, self.dataset_dict = build_dataloader(self.experiment_spec, mode="eval")
            self.class_dict = self.dataset_dict["query"].class_dict
        elif stage == 'predict':
            _, _, _, self.dataset_dict = build_dataloader(self.experiment_spec, mode="inference")
            self.class_dict = self.dataset_dict["gallery"].class_dict
        else:
            pass

    def train_dataloader(self):
        """Builds the dataloader for training.

        Returns:
            train_loader (torch.utils.data.Dataloader): Traininig Data.
        """
        return self.train_loader

    def test_dataloader(self):
        """Builds the dataloader for testing.

        Returns:
            test_loader (torch.utils.data.Dataloader): Testing Data.
        """
        # In reality, this dataloader isn't used but is necessary for Trainer.test() to not error
        return self.test_loader

    def predict_dataloader(self):
        """Builds the dataloader for inference.

        Returns:
            predict_loader (torch.utils.data.Dataloader): Inference Data.
        """
        predict_loader = build_inference_dataloader(self.experiment_spec)
        return predict_loader
