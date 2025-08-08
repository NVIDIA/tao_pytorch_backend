# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Transforms for Sparse4D dataset."""

import numpy as np
import torch
from PIL import Image
from typing import Dict
import h5py


class LoadMultiViewImageFromFiles:
    """Load multi-view images from files."""

    def __init__(self, to_float32=False, color_type="unchanged", h5_file=False):
        """Initialize transform.

        Args:
            to_float32: Whether to convert to float32
            color_type: Color type (unchanged, color, etc.)
            h5_file: Whether to load h5 file
        """
        self.to_float32 = to_float32
        self.color_type = color_type
        self.h5_file = h5_file

    def __call__(self, results: Dict) -> Dict:
        """Call function to load multi-view images.

        Args:
            results: Dict with image filenames

        Returns:
            Dict with loaded images
        """
        filename = results["img_filename"]
        # Load images (shape: h, w, c, num_views)
        if self.h5_file:
            img = []
            for name in filename:
                with h5py.File(name[0], "r") as f:
                    img.append(f[name[1]][:])
            img = np.stack(img, axis=-1)
        else:
            img = np.stack([Image.open(name).convert('RGB') for name in filename], axis=-1)

        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        # Unravel to list, each image has shape (h, w, c)
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = [img[..., i].shape for i in range(img.shape[-1])]
        results["ori_shape"] = results["img_shape"]

        # Set default values
        results["pad_shape"] = results["img_shape"]
        results["scale_factor"] = 1.0

        return results


class LoadDepthMap:
    """Load depth maps from files."""

    def __init__(self, max_depth=100, default_shape=(1080, 1920), h5_file=False):
        """Initialize transform.

        Args:
            max_depth: Maximum depth value
            default_shape: Default shape of depth map
            h5_file: Whether to load h5 file
        """
        self.max_depth = max_depth
        self.default_shape = default_shape
        self.h5_file = h5_file

    def __call__(self, results: Dict) -> Dict:
        """Call function to load depth maps.

        Args:
            results: Dict with depth map filenames

        Returns:
            Dict with loaded depth maps
        """
        if "depth_map_filename" not in results:
            return results

        filename = results["depth_map_filename"]
        # Load depth maps (shape: h, w, num_views)
        depth_maps = []

        if self.h5_file:
            depths = []
            for name in filename:
                with h5py.File(name[0], "r") as f:
                    depths.append(f[name[1]][:])
            depths = np.stack(depths, axis=-1)
        else:
            for name in filename:
                # Open depth map as grayscale image
                if name is None:
                    depth = np.ones(self.default_shape) * -1
                else:
                    depth = Image.open(name)
                    depth = np.array(depth)
                depth_maps.append(depth)

            depths = np.stack(depth_maps, axis=-1)
        depths = depths.astype(np.float32)
        depths /= 1000.  # Convert from mm to m
        depths = np.clip(depths, 0.1, self.max_depth)

        # Convert to list of depth maps
        gt_depth = []
        for i, _ in enumerate(results["lidar2img"]):
            gt_depth.append(depths[..., i])

        results["gt_depth"] = gt_depth
        return results


class InstanceNameFilter:
    """Filter instances by class names."""

    def __init__(self, classes):
        """Initialize transform.

        Args:
            classes: List of class names to keep
        """
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, results: Dict) -> Dict:
        """Filter objects by class names.

        Args:
            results: Dict with instances

        Returns:
            Dict with filtered instances
        """
        if "gt_labels_3d" not in results:
            return results

        gt_labels_3d = results["gt_labels_3d"]
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d], dtype=np.bool_)

        results["gt_bboxes_3d"] = results["gt_bboxes_3d"][gt_bboxes_mask]
        results["gt_labels_3d"] = results["gt_labels_3d"][gt_bboxes_mask]

        if "instance_inds" in results:
            results["instance_inds"] = results["instance_inds"][gt_bboxes_mask]

        if "gt_visibility" in results:
            results["gt_visibility"] = results["gt_visibility"][gt_bboxes_mask]

        return results


class AICitySparse4DAdaptor:
    """Adapt data format for Sparse4D model."""

    def __call__(self, results: Dict) -> Dict:
        """Format data for Sparse4D model.

        Args:
            results: Dict with data

        Returns:
            Dict with formatted data
        """
        # Convert projection matrices
        if "lidar2img" in results:
            results["projection_mat"] = np.float32(np.stack(results["lidar2img"]))

        # Convert image dimensions
        if "img_shape" in results:
            results["image_wh"] = np.ascontiguousarray(
                np.array(results["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
            )

        # Process camera intrinsics
        if "cam_intrinsic" in results:
            results["cam_intrinsic"] = np.float32(np.stack(results["cam_intrinsic"]))
            results["focal"] = results["cam_intrinsic"][..., 0, 0]

        # Process instance IDs
        if "instance_inds" in results:
            results["instance_id"] = results["instance_inds"]

        # Process 3D bounding boxes
        if "gt_bboxes_3d" in results:
            # Normalize yaw angle
            results["gt_bboxes_3d"][:, 6] = self.limit_period(
                results["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )

            # Convert to tensor
            results["gt_bboxes_3d"] = torch.tensor(results["gt_bboxes_3d"]).float()

        # Process labels
        if "gt_labels_3d" in results:
            results["gt_labels_3d"] = torch.tensor(results["gt_labels_3d"]).long()

        # Process images
        imgs = [img.transpose(2, 0, 1) for img in results["img"]]  # HWC -> CHW
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        results["img"] = torch.tensor(imgs)
        return results

    @staticmethod
    def limit_period(val: np.ndarray, offset: float = 0.5, period: float = np.pi) -> np.ndarray:
        """Limit angle to the range of [-offset * period, (1-offset) * period].

        Args:
            val: Angle in radians
            offset: Offset for the periodic boundary
            period: Period for the angle

        Returns:
            Angle after limiting to a period
        """
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val

    @staticmethod
    def limit_period_torch(val: torch.Tensor, offset: float = 0.5, period: float = np.pi) -> torch.Tensor:
        """PyTorch version of limit_period.

        Args:
            val: Angle in radians
            offset: Offset for the periodic boundary
            period: Period for the angle

        Returns:
            Angle after limiting to a period
        """
        return val - torch.floor(val / period + offset) * period


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms):
        """Initialize transform.

        Args:
            transforms: List of transforms to apply
        """
        self.transforms = transforms

    def __call__(self, results: Dict) -> Dict:
        """Apply all transforms sequentially.

        Args:
            results: Dict with data

        Returns:
            Dict with transformed data
        """
        for t in self.transforms:
            results = t(results)
            if results is None:
                return None
        return results
