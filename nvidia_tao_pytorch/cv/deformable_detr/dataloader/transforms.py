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

""" Transformation for people transformer."""

import PIL
import torch
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F

from nvidia_tao_pytorch.cv.deformable_detr.utils.box_ops import box_xyxy_to_cxcywh


def build_transforms(augmentation_config, experiment_config=None, dataset_mode='train'):
    """Build Augmentations.

    Args:
        augmentation_config (dict): augmentation configuration.
        experiment_config (dict): experiment configuration.
        dataset_mode (str): data mode (train, val, eval, infer).

    Returns:
        transforms (Compose): Final built transforms.

    Raises:
        If dataset_mode is set to other than given options (train, val, eval, infer), the code will raise the value error.
    """
    input_mean = list(augmentation_config["input_mean"])
    input_std = list(augmentation_config["input_std"])
    scales = list(augmentation_config["scales"])
    ranom_resize_max_size = augmentation_config["random_resize_max_size"]
    test_random_size = augmentation_config["test_random_resize"]
    train_random_sizes = list(augmentation_config["train_random_resize"])
    train_random_crop_min = augmentation_config["train_random_crop_min"]
    train_random_crop_max = augmentation_config["train_random_crop_max"]
    flip_prob = min(1.0, augmentation_config["horizontal_flip_prob"])
    fixed_padding = augmentation_config["fixed_padding"]

    normalize = Compose([
        ToTensor(),
        Normalize(input_mean, input_std)
    ])

    # Fixed Padding is applied to prevent memory leak
    # It nees to be applied prior to normalize transform
    # Padding has same effect as the collate_fn as only the original image is passed as the size
    if dataset_mode == 'train':
        if fixed_padding:
            transforms = Compose([
                RandomHorizontalFlip(flip_prob),
                RandomSelect(
                    RandomResize(scales, max_size=ranom_resize_max_size),
                    Compose([
                        RandomResize(train_random_sizes),
                        RandomSizeCrop(train_random_crop_min, train_random_crop_max),
                        RandomResize(scales, max_size=ranom_resize_max_size),
                    ])
                ),
                normalize,
                FixedPad(sorted(scales)[-1], ranom_resize_max_size),
            ])
        else:
            transforms = Compose([
                RandomHorizontalFlip(flip_prob),
                RandomSelect(
                    RandomResize(scales, max_size=ranom_resize_max_size),
                    Compose([
                        RandomResize(train_random_sizes),
                        RandomSizeCrop(train_random_crop_min, train_random_crop_max),
                        RandomResize(scales, max_size=ranom_resize_max_size),
                    ])
                ),
                normalize,
            ])
    elif dataset_mode in ('val', 'eval', 'infer'):
        if fixed_padding:
            transforms = Compose([
                RandomResize([test_random_size], max_size=ranom_resize_max_size),
                normalize,
                FixedPad(test_random_size, ranom_resize_max_size),
            ])
        else:
            transforms = Compose([
                RandomResize([test_random_size], max_size=ranom_resize_max_size),
                normalize,
            ])
    else:
        raise ValueError('There are only train, val, eval, and infer options in dataset_mode.')

    return transforms


def crop(image, target, region):
    """Crop image.

    Args:
        image (PIL.Image): loaded image.
        target (dict): loaded target.
        region (tuple): region to crop.

    Returns:
        (cropped_image, taret): cropped image and processed target based on the cropped image

    """
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    target["size"] = torch.tensor([h, w])

    fields = ["labels"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("boxes")

    # remove elements for which the boxes that have zero area
    if "boxes" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    """Horizontal Flip.

    Args:
        image (PIL.image): loaded image.
        target (dict): loaded target.

    Returns:
        (flipped_image, taret): flipped image and processed target based on the flipped image.
    """
    flipped_image = F.hflip(image)

    w, _ = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    return flipped_image, target


def resize(image, target, size, max_size=None):
    """Resize the image.

    Args:
        image (PIL.Image): loaded image.
        target (dict): loaded target.
        size (int / tuple): size to resize, size can be min_size (scalar) or (w, h) tuple.
        max_size (int): maximum size to resize.

    Returns:
        (rescaled_image, taret): rescaled image and processed target based on the rescaled image.
    """
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        """ get size with aspect ratio """
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        """ get size to resize """
        if isinstance(size, (list, tuple)):
            return_size = size[::-1]
        else:
            return_size = get_size_with_aspect_ratio(image_size, size, max_size)
        return return_size

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


def pad(image, target, padding):
    """Padding the image on the bottom right corners.

    Args:
        image (PIL.Image): loaded image.
        target (dict): loaded target.
        padding (tuple): size to pad.

    Returns:
        (padded_image, taret): padded image and processed target based on the padded image.
    """
    # zero padding
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()

    # We pass size as pre-padded image so that collate_fn can overwrite the
    # transform-padded region too
    if isinstance(image, torch.Tensor):
        target["size"] = image.shape[1:]
    else:
        target["size"] = torch.tensor(image.size[::-1])

    return padded_image, target


class RandomCrop(object):
    """Random Crop class."""

    def __init__(self, size):
        """Initialize the RandomCrop Class.

        Args:
            size (tuple): size to perform random crop
        """
        self.size = size

    def __call__(self, img, target):
        """Call RandomCrop.

        Args:
            image (PIL.Image): Pillow Image.
            target (dict): Annotations.

        Returns:
            image (PIL.Image): Cropped Image.
            target (dict): Cropped Annotations.
        """
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    """Random Size Crop class."""

    def __init__(self, min_size: int, max_size: int):
        """Initialize the RandomCrop Class.

        Args:
            min_size (int): minimum size to perform random crop.
            max_size (int): maximum size to perform random crop.
        """
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        """Call RandomSizeCrop.

        Args:
            image (PIL.Image): Pillow Image.
            target (dict): Annotations.

        Returns:
            image (PIL.Image): Cropped Image.
            target (dict): Cropped Annotations.
        """
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    """Center Crop class."""

    def __init__(self, size):
        """Initialize the CenterCrop Class.

        Args:
            size (tuple): size to perform center crop.
        """
        self.size = size

    def __call__(self, img, target):
        """Call CenterCrop.

        Args:
            image (PIL.Image): Pillow Image.
            target (dict): Annotations.

        Returns:
            image (PIL.Image): Cropped Image.
            target (dict): Cropped Annotations.
        """
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    """Random Horizontal Flip class"""

    def __init__(self, p=0.5):
        """Initialize the RandomHorizontalFlip Class.

        Args:
            p (float): probability to perform random horizontal flip.
        """
        self.p = p

    def __call__(self, img, target):
        """Call RandomHorizontalFlip.

        Args:
            image (PIL.Image): Pillow Image.
            target (dict): Annotations.

        Returns:
            image (PIL.Image): Flipped Image.
            target (dict): Flipped Annotations.
        """
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    """Random Resize class."""

    def __init__(self, sizes, max_size=None):
        """Initialize the RandomResize Class.

        Args:
            size (list): size to perform random resize.
            max_size (int): maximum size to perform random resize.
        """
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        """Call RandomResize.

        Args:
            image (PIL.Image): Pillow Image.
            target (dict): Annotations.

        Returns:
            image (PIL.Image): Resized Image.
            target (dict): Resized Annotations.
        """
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class FixedResize(object):
    """Fixed Size Resize class."""

    def __init__(self, sizes):
        """Initialize the FixedResize Class.

        Args:
            sizes (list): size to perform random resize.
        """
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes

    def __call__(self, img, target=None):
        """Call FixedResize.

        Args:
            image (PIL.Image): Pillow Image.
            target (dict): Annotations.

        Returns:
            image (PIL.Image): Resized Image.
            target (dict): Resized Annotations.
        """
        return resize(img, target, self.sizes, None)


class RandomPad(object):
    """Random Pad class."""

    def __init__(self, max_pad):
        """Initialize the RandomPad Class.

        Args:
            max_pad (int): max padding size.
        """
        self.max_pad = max_pad

    def __call__(self, img, target):
        """Call RandomPad.

        Args:
            image (PIL.Image): Pillow Image.
            target (dict): Annotations.

        Returns:
            image (PIL.Image): Padded Image.
            target (dict): Padded Annotations.
        """
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class FixedPad(object):
    """Fixed Pad class."""

    def __init__(self, target_min, target_max):
        """Initialize the FixedPad Class.

        Args:
            target_width (int): padding size.
            target_height (int): padding size.
        """
        self.target_min = target_min
        self.target_max = target_max

    def __call__(self, img, target):
        """Call FixedPad.

        Args:
            image (PIL.Image): Pillow Image.
            target (dict): Annotations.

        Returns:
            image (PIL.Image): Padded Image.
            target (dict): Padded Annotations.
        """
        height, width = target['size']
        if height > width:
            pad_x = self.target_min - width
            pad_y = self.target_max - height
        else:
            pad_x = self.target_max - width
            pad_y = self.target_min - height
        tmp = pad(img, target, (pad_x, pad_y))
        return tmp


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2.
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        """Initialize the RandomSelect Class.

        Args:
            transforms1 (object): given transform to select.
            transforms2 (object): given transform to select.
            p (float): probability to select between transform 1 and 2.
        """
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        """Call RandomSelect.

        Args:
            image (PIL.Image): Pillow Image.
            target (dict): Annotations.

        Returns:
            image (PIL.Image): Image.
            target (dict): Annotations.
        """
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    """Convert PIL.Image to torch.Tensor"""

    def __call__(self, img, target):
        """Call ToTensor.

        Args:
            image (PIL.Image): Pillow Image.
            target (dict): Annotations.

        Returns:
            image (torch.Tensor): Image Tensor.
            target (dict): Annotations.
        """
        return F.to_tensor(img), target


class RandomErasing(object):
    """Random Erasing class."""

    def __init__(self, *args, **kwargs):
        """Initialize the RandomErasing Class."""
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        """Call RandomErasing.

        Args:
            image (PIL.Image): Pillow Image.
            target (dict): Annotations.

        Returns:
            image (PIL.Image): Randomly erased Image.
            target (dict): Randomly erased Annotations.
        """
        return self.eraser(img), target


class Normalize(object):
    """ Normalize class """

    def __init__(self, mean, std):
        """Initialize the Normalize Class.

        Args:
            mean (list): mean value to normalize.
            std (list): standard deviation value to normalize.
        """
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        """Call Normalize.

        Args:
            image (PIL.Image): Pillow Image.
            target (dict): Annotations.

        Returns:
            image (torch.Tensor): Normalized Tensor.
            target (dict): Normalized Annotations.
        """
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    """Compose class."""

    def __init__(self, transforms):
        """Initialize the Compose Class.

        Args:
            transforms (list): transform list to compose.
        """
        self.transforms = transforms

    def __call__(self, image, target):
        """Call Compose.

        Args:
            image (torch.Tensor): Image in Tensor.
            target (dict): Annotations.

        Returns:
            image (torch.Tensor): Composed Tensor.
            target (dict): Composed Annotations.
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        """ repr """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
