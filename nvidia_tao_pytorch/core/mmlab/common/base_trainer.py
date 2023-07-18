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

"""Base MMCV Trainer Class."""

import torch
from mmcv.parallel import MMDistributedDataParallel
from abc import ABC, abstractmethod


class MMCVTrainer(ABC):
    """MMCV Base Trainer"""

    def __init__(self,
                 dataset,
                 model,
                 timestamp=None,
                 meta=None,
                 result_dir=None,
                 experiment_spec=None):
        """Init Function.
        Args:
            dataset (dataset instance): Imagenet Dataset type instance.
            model (nn.Module): PyT model instance.
            meta (Dict): Contains the env variables.
            result_dir (str): Path to the results dir.
            experiment_spec (Dict): Contains the hydra exp config parameters.

        """
        self.model = model
        self.dataset = dataset
        self.timestamp = timestamp
        self.result_dir = result_dir
        self.cfg = experiment_spec
        self.model_cfg = experiment_spec["model"]
        self.train_cfg = experiment_spec["train"]["train_config"]
        self.dataset_cfg = experiment_spec["dataset"]
        self.meta = meta
        self.evaluation_cfg = experiment_spec["train"]["train_config"]["evaluation"]

    @abstractmethod
    def set_up_data_loaders(self):
        """Function to generate dataloaders."""
        raise NotImplementedError(
            "Base Trainer doesn't implement data loader instantiation."
        )

    @abstractmethod
    def validate_runner(self):
        """Function to Add validation hook to training"""
        raise NotImplementedError(
            "Base Trainer doesn't implement validation for runner instantiation."
        )

    def set_up_trainer(self):
        """ Set up the end-end trainer"""
        self.data_loaders = self.set_up_data_loaders()
        self.model = self.set_up_model()
        self.runner = self.set_up_runner()
        if self.train_cfg["validate"]:
            # Add the validation hook to the runner if validate is True
            self.validate_runner()

    def set_up_model(self):
        """Function To Set Up Model"""
        # put model on gpus
        find_unused_parameters = self.train_cfg["find_unused_parameters"]
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        self.model = MMDistributedDataParallel(
            self.model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

        return self.model

    @abstractmethod
    def set_up_runner(self):
        """Function to Build the Runner."""
        raise NotImplementedError(
            "Base Trainer doesn't implement data loader instantiation."
        )

    def fit(self):
        """Runner Fit to Start the training."""
        self.runner.run(self.data_loaders, workflow=[('train', 1)])
