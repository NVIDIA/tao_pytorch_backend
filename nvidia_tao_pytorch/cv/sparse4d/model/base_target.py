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

"""Base Target for Sparse4D."""
from abc import ABC, abstractmethod


__all__ = ["BaseTargetWithDenoising"]


class BaseTargetWithDenoising(ABC):
    """Base target with denoising."""

    def __init__(self, num_dn_groups=0, num_temp_dn_groups=0):
        """Initialize the BaseTargetWithDenoising.
        Args:
            num_dn_groups: Number of denoising groups
            num_temp_dn_groups: Number of temporary denoising groups
        """
        super(BaseTargetWithDenoising, self).__init__()
        self.num_dn_groups = num_dn_groups
        self.num_temp_dn_groups = num_temp_dn_groups
        self.dn_metas = None

    @abstractmethod
    def sample(self, cls_pred, box_pred, cls_target, box_target):
        """
        Perform Hungarian matching between predictions and ground truth,
        returning the matched ground truth corresponding to the predictions
        along with the corresponding regression weights.
        """

    def get_dn_anchors(self, cls_target, box_target, *args, **kwargs):
        """
        Generate noisy instances for the current frame, with a total of
        'self.num_dn_groups' groups.
        """
        return None

    def update_dn(self, instance_feature, anchor, *args, **kwargs):
        """
        Insert the previously saved 'self.dn_metas' into the noisy instances
        of the current frame.
        """

    def cache_dn(
        self,
        dn_instance_feature,
        dn_anchor,
        dn_cls_target,
        valid_mask,
        dn_id_target,
    ):
        """
        Randomly save information for 'self.num_temp_dn_groups' groups of
        temporal noisy instances to 'self.dn_metas'.
        """
        if self.num_temp_dn_groups < 0:
            return
        self.dn_metas = dict(dn_anchor=dn_anchor[:, : self.num_temp_dn_groups])
