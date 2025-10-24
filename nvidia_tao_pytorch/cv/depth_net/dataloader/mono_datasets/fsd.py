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

"""Dataset Class for FSD data."""

from nvidia_tao_pytorch.cv.depth_net.dataloader.utils.frame_utils import read_gt_fsdv3
from nvidia_tao_pytorch.cv.depth_net.dataloader.mono_datasets.base_relative_mono import BaseRelativeMonoDataset


class FSD(BaseRelativeMonoDataset):
    """Dataset class for FSD, providing ground truth in disparity format."""

    def read_gt_depth(self, disp_path):
        """Read FSD ground truth disparity and mask data.

        Args:
            disp_path (str): path to the disparity map.

        Returns:
            depth (np.ndarray): depth map.
        """
        return read_gt_fsdv3(disp_path, normalize_depth=self.normalize_depth)
