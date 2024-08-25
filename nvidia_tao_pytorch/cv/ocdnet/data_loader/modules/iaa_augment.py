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
#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
"""Imgaug augment module."""
import numpy as np
import imgaug
import imgaug.augmenters as iaa


class AugmenterBuilder(object):
    """Augmenter Builder."""

    def __init__(self):
        """Initialize."""
        pass

    def build(self, args, root=True):
        """Build augmenter."""
        if args is None or len(args) == 0:
            return None
        if isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            return getattr(iaa, args[0])(*[self.convert_object(a) for a in args[1:]])
        if isinstance(args, dict):
            cls = getattr(iaa, args['type'])
            return cls(**{k: self.convert_object(v) for k, v in args['args'].items()})
        raise RuntimeError('unknown augmenter arg: ' + str(args))

    def convert_object(self, obj):
        """Convert the object data type."""
        if isinstance(obj, list):
            return tuple(obj)
        if isinstance(obj, dict):
            return self.build(obj, root=False)
        return obj


class IaaAugment():
    """Imgaug augment class."""

    def __init__(self, augmenter_args):
        """Initialize."""
        self.augmenter_args = augmenter_args
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def __call__(self, data):
        """Imgaug augmentation."""
        image = data['img']
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            data['img'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        return data

    def may_augment_annotation(self, aug, data, shape):
        """Augment annotation."""
        if aug is None:
            return data

        line_polys = []
        for poly in data['text_polys']:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        data['text_polys'] = np.array(line_polys)
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        """Augment poly."""
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly
