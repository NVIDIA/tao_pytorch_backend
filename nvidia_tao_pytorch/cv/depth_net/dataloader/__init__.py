# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""DepthNet dataloader module."""

from nvidia_tao_pytorch.cv.depth_net.dataloader.pl_mono_data_module import MonoDepthNetDataModule
from nvidia_tao_pytorch.cv.depth_net.dataloader.pl_stereo_data_module import StereoDepthNetDataModule

_pl_data_modules = {'MonoDataset': MonoDepthNetDataModule,
                    'StereoDataset': StereoDepthNetDataModule}


def build_pl_data_module(dataset_config):
    """Build lightning data_module given the dataset_config from spec file.

    Args:
        dataset_config (dict): dataset configuration.

    Returns:
        pl_data_module (class): lightning data module.
    """
    return _pl_data_modules[dataset_config.dataset_name](dataset_config)
