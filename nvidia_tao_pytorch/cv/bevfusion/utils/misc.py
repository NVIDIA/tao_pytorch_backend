# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BEVFusion Misc utility Function"""


def sanity_check(config):
    """sanity check for config setup"""
    if (config['per_sequence'] is False and config['sequence_list'] is None) or \
       (config['per_sequence'] is True and config['sequence_list'] is not None):
        return True
    per_sequence = config['per_sequence']
    seq_list = config['sequence_list']
    raise ValueError(f'you must specify both per_sequence and sequence_list in the config file. \
                     Currently per_sequence is {per_sequence} and sequence_list is {seq_list}')


def prepare_origin_per_dataset(config):
    """prepare origin and yaw_dim based on config file"""
    yaw_dim = -1
    is_synthetic = False
    if config['dataset']['type'] == 'KittiPersonDataset':
        origin = (0.5, 1.0, 0.5)
        yaw_dim = 1  # y-axis
    elif config['dataset']['type'] == 'TAO3DDataset':  # GT in Lidar Space
        origin = (0.5, 0.5, 0.5)
        yaw_dim = 2  # z-axis
    elif config['dataset']['type'] == 'TAO3DSyntheticDataset':
        origin = (0.5, 0.5, 0.0)
        yaw_dim = -1  # use all rotations
        is_synthetic = True
    else:
        print('overwriting origin with given value in config file for new dataset')
        origin = config['dataset']['origin']
    return origin, yaw_dim, is_synthetic
