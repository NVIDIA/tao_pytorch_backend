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

"""Generate LIDAR labels."""
import os
import argparse

import numpy as np
from tqdm import tqdm

from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils.object3d_kitti import (
    get_objects_from_label
)
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils.calibration_kitti import (
    Calibration
)


def parse_args():
    """Argument Parser."""
    parser = argparse.ArgumentParser("Convert camera label to LiDAR label.")
    parser.add_argument(
        "-l", "--label_dir",
        type=str, required=True,
        help="Camera label directory."
    )
    parser.add_argument(
        "-c", "--calib_dir",
        type=str, required=True,
        help="Calibration file directory"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str, required=True,
        help="Output LiDAR label directory"
    )
    return parser.parse_args()


def generate_lidar_labels(label_dir, calib_dir, output_dir):
    """Generate LiDAR labels from KITTI Camera labels."""
    if os.path.isdir(label_dir):
        for lab in tqdm(os.listdir(label_dir)):
            lab_file = os.path.join(label_dir, lab)
            obj_list = get_objects_from_label(lab_file)
            calib_file = os.path.join(calib_dir, lab)
            calib = Calibration(calib_file)
            loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
            loc_lidar = calib.rect_to_lidar(loc)
            # update obj3d.loc
            with open(os.path.join(output_dir, lab), "w") as lf:
                for idx, lc in enumerate(loc_lidar):
                    # bottom center to 3D center
                    obj_list[idx].loc = (lc + np.array([0., 0., obj_list[idx].h / 2.]))
                    # rotation_y to rotation_z
                    obj_list[idx].ry = -np.pi / 2. - obj_list[idx].ry
                    lf.write(obj_list[idx].to_kitti_format())
                    lf.write('\n')


if __name__ == "__main__":
    args = parse_args()
    label_dir = expand_path(args.label_dir)
    calib_dir = expand_path(args.calib_dir)
    output_dir = expand_path(args.output_dir)
    generate_lidar_labels(label_dir, calib_dir, output_dir)
