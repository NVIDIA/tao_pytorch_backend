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

"""BEVFusion loading functions """

from typing import List, Optional, Union
import numpy as np
import copy

import mmcv
from mmcv.transforms import BaseTransform
import mmengine
from mmengine.fileio import get
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import get_points_type
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles


@TRANSFORMS.register_module()
class TAOLoadPointsFromFile(BaseTransform):
    """Load Points From File.
    Required Keys:
    - lidar_points (dict)
    - lidar_path (str)
    Added Keys:
    - points (np.float32)
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        """
        Args:
            coord_type (str): The type of coordinates of points cloud.
                Available options includes:

                - 'LIDAR': Points in LiDAR coordinates.
                - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
                - 'CAMERA': Points in camera coordinates.
            load_dim (int): The dimension of the loaded points. Defaults to 6.
            use_dim (list[int] | int): Which dimensions of the points to use.
                Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
                or use_dim=[0, 1, 2, 3] to use the intensity dimension.
            shift_height (bool): Whether to use shifted height. Defaults to False.
            use_color (bool): Whether to use color features. Defaults to False.
            norm_intensity (bool): Whether to normlize the intensity. Defaults to
                False.
            norm_elongation (bool): Whether to normlize the elongation. This is
                usually used in Waymo dataset.Defaults to False.
            backend_args (dict, optional): Arguments to instantiate the
                corresponding backend. Defaults to None.
        """
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmengine.check_file_exist(pts_filename)
        if pts_filename.endswith('.npy'):
            points = np.load(pts_filename, allow_pickle=True)
        elif pts_filename.endswith('.bin'):
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        else:
            points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_file_path = results['lidar_points']['lidar_path']
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]

        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, \
                f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = {'height': 3}

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = {}
            attribute_dims.update({'color': [points.shape[1] - 3, points.shape[1] - 2, points.shape[1] - 1,]})

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)

        results['points'] = points

        return results


@TRANSFORMS.register_module()
class BEVFusionLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load single view channel images from a list of separate channel files.

    ``TAO3DLoadSingleViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'unchanged',
                 backend_args: Optional[dict] = None,
                 num_views: int = 1,
                 num_ref_frames: int = -1,
                 test_mode: bool = False,
                 set_default_scale: bool = True) -> None:
        """
        Args:
            to_float32 (bool): Whether to convert the img to float32.
                Defaults to False.
            color_type (str): Color type of the file. Defaults to 'unchanged'.
            backend_args (dict, optional): Arguments to instantiate the
                corresponding backend. Defaults to None.
            num_views (int): Number of view in a frame. Defaults to 5.
            num_ref_frames (int): Number of frame in loading. Defaults to -1.
            test_mode (bool): Whether is test mode in loading. Defaults to False.
            set_default_scale (bool): Whether to set default scale.
                Defaults to True.
        """
        super().__init__(to_float32=to_float32,
                         color_type=color_type,
                         backend_args=backend_args,
                         num_views=num_views,
                         num_ref_frames=num_ref_frames,
                         test_mode=test_mode,
                         set_default_scale=set_default_scale)

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename, cam2img, lidar2cam, cam2lidar, lidar2img = [], [], [], [], []
        for _, cam_item in results['images'].items():
            if 'img_path' not in cam_item:
                continue
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            lidar2cam_array = np.array(cam_item['lidar2cam']).astype(np.float32)

            cam2lidar.append(np.linalg.inv(lidar2cam_array))
            if np.array(cam_item['cam2img']).shape == (4, 4):
                cam2img_array = np.array(cam_item['cam2img']).astype(np.float32)
            else:  # extend cam2img matrix in case 3x3 is provided. need to check
                cam2img_array = np.eye(4).astype(np.float32)
                cam2img_array[:3, :3] = np.array(cam_item['cam2img']).astype(np.float32)
            cam2img.append(cam2img_array)

            if 'lidar2img' in cam_item:
                lidar2img.append(cam_item['lidar2img'])
            else:
                lidar2img.append(cam2img_array @ lidar2cam_array)
        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)  # (view, 4, 4)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)  # (view, 4, 4)
        results['cam2lidar'] = np.stack(cam2lidar, axis=0)  # (view, 4, 4)
        results['lidar2img'] = np.stack(lidar2img, axis=0)  # (view, 4, 4)
        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type,
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]

        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = np.array([1.0, 1.0, 1.0, 1.0],
                                               dtype=np.float32)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = {
            'mean': np.zeros(num_channels, dtype=np.float32),
            'std': np.ones(num_channels, dtype=np.float32),
            'to_rg': False}
        results['num_views'] = self.num_views
        return results
