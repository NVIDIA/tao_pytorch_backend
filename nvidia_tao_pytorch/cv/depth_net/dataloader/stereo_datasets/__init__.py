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

"""Init function for defining dataset classes"""

from torch.utils.data import ConcatDataset
from nvidia_tao_pytorch.cv.depth_net.dataloader.stereo_datasets.stereo_dataset import FSD, IsaacRealDataset, Crestereo, Middlebury, Eth3d, Kitti, GenericDataset

STEREO_DATASETS = {
    'fsd': FSD,
    'isaacrealdataset': IsaacRealDataset,
    'crestereo': Crestereo,
    'middlebury': Middlebury,
    'eth3d': Eth3d,
    'kitti': Kitti,
    'genericdataset': GenericDataset
}


def build_stereo_dataset(data_sources, transform, max_disparity=None):
    """Load Monocular Relative Depth Dataset.

    Args:
        data_sources (str): list of different data sources.
        transforms (dict): augmentations to apply.
        min_depth (float): minimum depth value.
        max_depth (float): maximum depth value.
        normalize_depth (bool): whether to normalize the depth.
    """
    if type(data_sources).__name__ == "DictConfig":
        data_sources = [data_sources]

    dataset_list = []
    for data_source in data_sources:
        data_file = data_source.data_file
        dataset_name = data_source.dataset_name
        model_cls = STEREO_DATASETS[dataset_name.lower()]
        dataset_list.append(model_cls(data_file, transform, max_disparity))
        if len(dataset_list) > 1:
            train_dataset = ConcatDataset(dataset_list)
        else:
            train_dataset = dataset_list[0]
    return train_dataset
