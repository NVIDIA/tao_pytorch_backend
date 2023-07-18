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

"""Dataset convert script for PointPillars."""
import os

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.datasets.general.pc_dataset import create_pc_infos

import argparse
from easydict import EasyDict
from pathlib import Path
import yaml


def parse_args(args=None):
    """Argument Parser."""
    parser = argparse.ArgumentParser(description="General point cloud dataset converter.")
    parser.add_argument("--cfg_file", "-c", type=str, help="Config file.")
    return parser.parse_known_args(args)[0]


if __name__ == "__main__":
    args = parse_args()
    cfg_file = expand_path(args.cfg_file)
    with open(cfg_file) as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))  # nosec
    names = cfg.dataset.class_names
    data_path = cfg.dataset.data_path
    results_dir = cfg.results_dir

    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting PointPillars dataset convert")
    try:
        create_pc_infos(
            dataset_cfg=cfg.dataset,
            class_names=names,
            data_path=Path(data_path),
            save_path=Path(data_path),
            status_logging=status_logging
        )
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Dataset convert finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Dataset convert was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
