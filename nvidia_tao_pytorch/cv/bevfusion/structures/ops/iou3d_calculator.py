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
# Modified from mmmdet3d. https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/

"""BEVFusion iou calculator functions"""

from mmdet3d.registry import TASK_UTILS
from mmdet3d.structures.bbox_3d import get_box_type


@TASK_UTILS.register_module(force=True)
class MyBboxOverlaps3D(object):
    """3D IoU Calculator.

    Args:
        coordinate (str): The coordinate system, valid options are
            'camera', 'lidar', and 'depth'.
    """

    def __init__(self, coordinate):
        """Init"""
        assert coordinate in ['camera', 'lidar', 'depth']
        self.coordinate = coordinate

    def __call__(self, bboxes1, bboxes2, mode='iou'):
        """Calculate 3D IoU using cuda implementation.

        Note:
            This function calculate the IoU of 3D boxes based on their volumes.
            IoU calculator ``:class:BboxOverlaps3D`` uses this function to
            calculate the actual 3D IoUs of boxes.

        Args:
            bboxes1 (torch.Tensor): with shape (N, 7+C),
                (x, y, z, x_size, y_size, z_size, ry, v*).
            bboxes2 (torch.Tensor): with shape (M, 7+C),
                (x, y, z, x_size, y_size, z_size, ry, v*).
            mode (str): "iou" (intersection over union) or
                iof (intersection over foreground).

        Return:
            torch.Tensor: Bbox overlaps results of bboxes1 and bboxes2
                with shape (M, N) (aligned mode is not supported currently).
        """
        return bbox_overlaps_3d(bboxes1, bboxes2, mode, self.coordinate)

    def __repr__(self):
        """str: return a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(coordinate={self.coordinate}'
        return repr_str


def bbox_overlaps_3d(bboxes1, bboxes2, mode='iou', coordinate='camera'):
    """Calculate 3D IoU using cuda implementation.

    Note:
        This function calculates the IoU of 3D boxes based on their volumes.
        IoU calculator :class:`BboxOverlaps3D` uses this function to
        calculate the actual IoUs of boxes.

    Args:
        bboxes1 (torch.Tensor): with shape (N, 9),
            (x, y, z, x_size, y_size, z_size, rx, ry, rz).
        bboxes2 (torch.Tensor): with shape (M, 9),
            (x, y, z, x_size, y_size, z_size, rx, ry, rz).
        mode (str): "iou" (intersection over union) or
            iof (intersection over foreground).
        coordinate (str): 'camera' or 'lidar' coordinate system.

    Return:
        torch.Tensor: Bbox overlaps results of bboxes1 and bboxes2
            with shape (M, N) (aligned mode is not supported currently).
    """
    assert bboxes1.size(-1) == bboxes2.size(-1) >= 7

    box_type, _ = get_box_type(coordinate)

    bboxes1 = box_type(bboxes1, box_dim=bboxes1.shape[-1])
    bboxes2 = box_type(bboxes2, box_dim=bboxes2.shape[-1])

    return bboxes1.overlaps(bboxes1, bboxes2, mode=mode)
