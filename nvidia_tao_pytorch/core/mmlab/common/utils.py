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

"""Utils Function"""

from mmcls.utils import collect_env
from mmcv.runner import get_dist_info, init_dist
import os
import glob


def set_env():
    """ Function to Set Environment """
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])

    meta['env_info'] = env_info
    return meta


def set_distributed(experiment_config, phase="train"):
    """ Set Distributed Params """
    rank, world_size = get_dist_info()
    # If distributed these env variables are set by torchrun
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)
    if "RANK" not in os.environ:
        os.environ['RANK'] = str(rank)
    if "WORLD_SIZE" not in os.environ:
        os.environ['WORLD_SIZE'] = str(world_size)
    if "MASTER_PORT" not in os.environ:
        os.environ['MASTER_PORT'] = str(experiment_config[phase]["exp_config"]["MASTER_PORT"])
    if "MASTER_ADDR" not in os.environ:
        os.environ['MASTER_ADDR'] = experiment_config[phase]["exp_config"]["MASTER_ADDR"]
    init_dist("pytorch", backend="nccl")


def get_latest_pth_model(results_dir):
    """Utility function to return the latest tlt model in a dir.
    Args:
        results_dir (str): Path to results dir.
    Returns:
        Returns the latest checkpoint.
    """
    files = list(filter(os.path.isfile, glob.glob(results_dir + "/*.pth")))
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x))
    latest_checkpoint = files[-1]
    if not os.path.isfile(latest_checkpoint):
        raise FileNotFoundError("Checkpoint file not found at {}").format(latest_checkpoint)
    return latest_checkpoint
