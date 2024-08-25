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

"""Split KITTI datset into train/val."""
import os
import sys
from nvidia_tao_pytorch.core.path_utils import expand_path


def split(list_file, lidar, label, output_lidar, output_label):
    """train/val split of the KITTI dataset."""
    with open(list_file) as lf:
        file_names = lf.readlines()
    file_names = [f.strip() for f in file_names]
    if os.path.isdir(lidar):
        for li in os.listdir(lidar):
            if li[:-4] in file_names:
                os.rename(os.path.join(lidar, li), os.path.join(output_lidar, li))
    if os.path.isdir(label):
        for la in os.listdir(label):
            if la[:-4] in file_names:
                os.rename(os.path.join(label, la), os.path.join(output_label, la))


if __name__ == "__main__":
    list_file = expand_path(sys.argv[1])
    lidar = expand_path(sys.argv[2])
    label = expand_path(sys.argv[3])
    output_lidar = expand_path(sys.argv[4])
    output_label = expand_path(sys.argv[5])

    split(list_file, lidar, label, output_lidar, output_label)
