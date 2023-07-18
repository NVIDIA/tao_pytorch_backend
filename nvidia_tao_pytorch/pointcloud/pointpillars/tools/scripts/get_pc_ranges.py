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

"""Get point cloud ranges."""
import os
import sys
import numpy as np
from nvidia_tao_pytorch.core.path_utils import expand_path


def calculate_pc_ranges(pc_path):
    """Get pointcloud data xyz ranges."""
    if os.path.isdir(pc_path):
        for idx, f in enumerate(os.listdir(pc_path)):
            pc_file = os.path.join(pc_path, f)
            xyz = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)
            if idx == 0:
                x_min = np.amin(xyz[:, 0])
                x_max = np.amax(xyz[:, 0])
                y_min = np.amin(xyz[:, 1])
                y_max = np.amax(xyz[:, 1])
                z_min = np.amin(xyz[:, 2])
                z_max = np.amax(xyz[:, 2])
            else:
                x_min = min(np.amin(xyz[:, 0]), x_min)
                x_max = max(np.amax(xyz[:, 0]), x_max)
                y_min = min(np.amin(xyz[:, 1]), y_min)
                y_max = max(np.amax(xyz[:, 1]), y_max)
                z_min = min(np.amin(xyz[:, 2]), z_min)
                z_max = max(np.amax(xyz[:, 2]), z_max)
        print("Pointcloud ranges: ", x_min, y_min, z_min, x_max, y_max, z_max)


if __name__ == "__main__":
    pc_path = expand_path(sys.argv[1])
    calculate_pc_ranges(pc_path)
