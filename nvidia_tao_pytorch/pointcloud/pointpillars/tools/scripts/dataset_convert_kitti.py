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

"""Dataset convert tool for KITTI dataset."""
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.datasets.kitti.kitti_dataset import create_kitti_infos

import argparse
from easydict import EasyDict
from pathlib import Path
import yaml
from nvidia_tao_pytorch.core.path_utils import expand_path


def parse_args(args=None):
    """Argument Parser."""
    parser = argparse.ArgumentParser(description="KITTI dataset converter.")
    parser.add_argument("--config_file", "-c", type=str, help="Dataset config file.")
    parser.add_argument("--names", "-n", type=str, help="Class names.")
    parser.add_argument("--data_path", "-d", type=str, help="KITTI data path.")
    parser.add_argument("--save_path", "-s", type=str, help="Output path.")
    return parser.parse_known_args(args)[0]


if __name__ == "__main__":
    args = parse_args()

    config_file = expand_path(args.config_file)
    data_path = expand_path(args.data_path)
    save_path = expand_path(args.save_path)

    with open(config_file) as f:
        dataset_cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    create_kitti_infos(
        dataset_cfg=dataset_cfg,
        class_names=args.names.strip().split(','),
        data_path=Path(data_path),
        save_path=Path(save_path)
    )
