# Original source taken from https://github.com/autonomousvision/stylegan-xl
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""StyleGAN Dataset. Streaming images and labels from datasets created with https://github.com/autonomousvision/stylegan-xl/blob/main/dataset_tool.py."""

import os
import numpy as np
import zipfile
import json
import torch
import copy
import io
from pathlib import Path
from PIL import Image
try:
    import pyspng
except ImportError:
    pyspng = None

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.sdg.stylegan_xl.utils import dnnlib


class Dataset(torch.utils.data.Dataset):
    """A custom dataset for loading images with coressponding class labels"""

    def __init__(
            self,
            name,                   # Name of the dataset.
            raw_shape,              # Shape of the raw image data (NCHW).
            max_size=None,          # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
            use_labels=False,       # Enable conditioning labels? False = label dimension is zero.
            xflip=False,            # Artificially double the size of the dataset via x-flips. Applied after max_size.
            random_seed=0,          # Random seed to use when applying max_size.
    ):
        """Initialize"""
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        self._base_raw_idx = copy.deepcopy(self._raw_idx)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def set_dyn_len(self, new_len):
        """
        Adjusts the dataset length dynamically by setting the internal index
        list to the first `new_len` elements of the base index list.
        """
        self._raw_idx = self._base_raw_idx[:new_len]

    def set_classes(self, cls_list):
        """Filters the dataset to only include the specified classes."""
        self._raw_labels = self._load_raw_labels()
        new_idcs = [self._raw_labels == cl for cl in cls_list]
        new_idcs = np.sum(np.vstack(new_idcs), 0)  # logical or
        new_idcs = np.where(new_idcs)  # find location
        self._raw_idx = self._base_raw_idx[new_idcs]
        assert all(sorted(cls_list) == np.unique(self._raw_labels[self._raw_idx]))
        logging.info(f"Training on the following classes: {cls_list}")

    def _get_raw_labels(self):
        """Get raw class label (integer label. NOT onehot label) numpy array from dataset.json."""
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        """Close the opend zip file."""
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        """Load a raw image without augmentation (flipping)."""
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        """Load raw class label (integer label. NOT onehot label) numpy array from dataset.json."""
        raise NotImplementedError

    def __getstate__(self):
        """Customize pickling to exclude _raw_labels from the serialized state."""
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        """Close the object, ignoring any exceptions."""
        try:
            self.close()
        except Exception:  # Catch all exceptions that inherit from Exception
            logging.warning("Closing object encountered an exception but was ignored.")

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self._raw_idx.size

    def __getitem__(self, idx):
        """Get a image and its coressponding class label from the dataset."""
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        """Get onehot class label."""
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        """Retrieve label and determine if flipping should be applied from a streaming index."""
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        """Get the name of the dataset."""
        return self._name

    @property
    def image_shape(self):
        """Get image shape."""
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        """Get image channels."""
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        """Get image resolution."""
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        """Get list of each label's shape."""
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        """Get label dimension."""
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        """Check if labels exist."""
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        """Check if onehot labels can be generated."""
        return self._get_raw_labels().dtype == np.int64


class ImageFolderDataset(Dataset):
    """Custom ImgageFolderDataset."""

    def __init__(
            self,
            path,                   # Path to directory or zip.
            resolution=None,        # Ensure specific resolution, None = highest available.
            **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        """Initialize"""
        self._path = path
        self._zipfile = None

        # Load images from directory or zip file
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        """Return the file extension of the given filename in lowercase."""
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        """Get the conetent of a zip file."""
        assert self._type == 'zip'
        if self._zipfile is None:
            f = open(self._path, 'rb')
            self.zip_content = f.read()
            f.close()
            self._zipfile = zipfile.ZipFile(io.BytesIO(self.zip_content), 'r')

        return self._zipfile

    def _open_file(self, fname):
        """Open image/label file."""
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        """Close the opend zip file."""
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        """Customize pickling to exclude _raw_labels from the serialized state."""
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        """Load a raw image without augmentation (flipping)."""
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        """Load raw class label (integer label. NOT onehot label) numpy array from dataset.json."""
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[str(Path(fname))] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
