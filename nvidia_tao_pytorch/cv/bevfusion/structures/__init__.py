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

"""BEVFusion structures module"""

from .bbox_3d import (TAOCameraInstance3DBoxes, TAOLiDARInstance3DBoxes,
                      project_cam2img, project_lidar2img,
                      get_rotation_matrix_3d, convert_cooridnates, get_box_type_tao3d)
from .ops import (MyBboxOverlaps3D, bbox_overlaps_3d)

# yapf: enable
__all__ = [
    'TAOCameraInstance3DBoxes', 'TAOLiDARInstance3DBoxes', 'project_cam2img', 'project_lidar2img',
    'get_rotation_matrix_3d', 'convert_cooridnates', 'MyBboxOverlaps3D', 'bbox_overlaps_3d', 'get_box_type_tao3d'
]
