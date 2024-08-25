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

"""Data preprocessor."""
from functools import partial

import numpy as np

from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:  # noqa: E722
    pass


class VoxelGeneratorWrapper():
    """Voxel Generator Wrapper."""

    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        """Initialize."""
        self.vsize_xyz = vsize_xyz
        self.coors_range_xyz = coors_range_xyz
        self.num_point_features = num_point_features
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.max_num_voxels = max_num_voxels

    def generate(self, points):
        """Genrate voxels from points."""
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            spconv_ver = 1
        except:  # noqa: E722
            try:
                from spconv.utils import VoxelGenerator
                spconv_ver = 1
            except:  # noqa: E722
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                spconv_ver = 2
        if spconv_ver == 1:
            _voxel_generator = VoxelGenerator(
                voxel_size=self.vsize_xyz,
                point_cloud_range=self.coors_range_xyz,
                max_num_points=self.max_num_points_per_voxel,
                max_voxels=self.max_num_voxels
            )
        else:
            _voxel_generator = VoxelGenerator(
                vsize_xyz=self.vsize_xyz,
                coors_range_xyz=self.coors_range_xyz,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=self.max_num_points_per_voxel,
                max_num_voxels=self.max_num_voxels
            )
        if spconv_ver == 1:
            voxel_output = _voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, "Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = _voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    """Data Processor."""

    def __init__(
        self, processor_configs,
        point_cloud_range, training,
        num_point_features
    ):
        """Initialize."""
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        self.num_point_features = num_point_features
        func_map = {key: getattr(DataProcessor, key) for key in vars(DataProcessor) if not key.startswith("__")}
        for cur_cfg in processor_configs:
            if cur_cfg.name in func_map:
                cur_processor = func_map[cur_cfg.name](self, config=cur_cfg)
                self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        """Mask points and boxes that are out of range."""
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.remove_outside_boxes and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        """Randomly shuffle points."""
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.shuffle[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        """Transform points to voxels."""
        if data_dict is None:
            voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.voxel_size,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.max_points_per_voxel,
                max_num_voxels=config.max_number_of_voxels[self.mode],
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.voxel_size)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.voxel_size
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        points = data_dict['points']
        voxel_output = voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        """Sample points."""
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)

            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
