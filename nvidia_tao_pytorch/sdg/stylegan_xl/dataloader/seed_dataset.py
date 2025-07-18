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

"""Common Seed Dataset for both StyleGAN and BigDatasetGAN"""

import torch


class SeedDataset(torch.utils.data.Dataset):
    """A custom dataset for loading integer seeds"""

    def __init__(self, seeds):
        """Initialize"""
        self.seeds = seeds

    def __len__(self):
        """Return the total number of seeds in the dataset."""
        return len(self.seeds)

    def __getitem__(self, idx):
        """Get a integer seed from the dataset."""
        seed = self.seeds[idx]
        return seed

# # Example list of seeds
# seeds = [42, 123, 256, 512, 1024]
# seed_dataset = SeedDataset(seeds)
