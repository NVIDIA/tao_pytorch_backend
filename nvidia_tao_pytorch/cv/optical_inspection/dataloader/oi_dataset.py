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

"""Optical Inspection dataset."""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from nvidia_tao_pytorch.core.tlt_logging import logging


class SiameseNetworkTRIDataset(Dataset):
    """Siamese Model Dataset Class"""

    def __init__(self, data_frame=None, transform=None,
                 input_data_path=None, train=True, data_config=None):
        """Initialize the SiameseNetworkTRIDataset.

        Args:
            data_frame (pd.DataFrame): The input data frame.
            transform (transforms.Compose): transformation to be applied to the input image samples.
            input_data_path (str): The path to the input data root directory.
            train (bool): Flag indicating whether the dataset is for training.
            data_config (OmegaConf.DictConf): Configuration for the dataset.
        """
        self.data_frame = data_frame
        self.transform = transform
        # self.data_path = data_path
        # self.input_images = data_config["validation_dataset"]["images_dir"]
        self.input_image_root = input_data_path
        self.train = train
        self.num_inputs = data_config["num_input"]
        self.concat_type = data_config["concat_type"]
        self.lighting = data_config["input_map"]
        self.grid_map = data_config["grid_map"]
        self.output_shape = data_config["output_shape"]
        self.ext = data_config["image_ext"]
        if self.concat_type == "grid":
            print("Using {} input types and {} type {} X {} for comparison ".format(
                self.num_inputs,
                self.concat_type,
                self.grid_map["x"],
                self.grid_map["y"]
            ))
        else:
            print("Using {} input types and {} type 1 X {} for comparison".format(
                self.num_inputs,
                self.concat_type,
                self.num_inputs
            ))
        # self.tensorBR = tensorBR
        self.ext = '.jpg'

    def get_absolute_image_path(self, prefix, light=None):
        """
        Get the absolute image path.

        Args:
            prefix (str): The prefix of the image path.
            light (str): The lighting condition suffix to be appended to image name.

        Returns:
            str: The absolute image path.
        """
        image_path = prefix
        if light:
            image_path += f"_{light}"
        image_path += self.ext
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file wasn't found at {image_path}")
        return image_path

    def __getitem__(self, index):
        """Get the item at a specific index."""
        img_tuple = self.data_frame.iloc[index, :]

        img0, img1 = [], []
        if self.lighting:
            for i, light in enumerate(self.lighting):
                img0.append(
                    Image.open(
                        self.get_absolute_image_path(
                            self.getComparePathsV1(img_tuple),
                            light
                        )
                    )
                )
                img1.append(
                    Image.open(
                        self.get_absolute_image_path(
                            self.getGoldenPathsV1(img_tuple),
                            light
                        )
                    )
                )
        else:
            img0.append(
                Image.open(
                    self.get_absolute_image_path(
                        self.getComparePathsV1(img_tuple)
                    )
                )
            )
            img1.append(
                Image.open(
                    self.get_absolute_image_path(
                        self.getGoldenPathsV1(img_tuple)
                    )
                )
            )
        if self.train:
            for i in range(len(img0)):
                img0[i] = img0[i].convert("RGB")
            for i in range(len(img1)):
                img1[i] = img1[i].convert("RGB")
            if self.transform is not None:
                img0T = [self.transform(img) for img in img0]
                img1T = [self.transform(img) for img in img1]
        else:
            if self.transform is not None:
                img0T = [self.transform(img) for img in img0]
                img1T = [self.transform(img) for img in img1]
        if self.concat_type == "grid" and int(self.num_inputs) % 2 == 0:
            img0 = self.get_grid_concat(img0T)
            img1 = self.get_grid_concat(img1T)
        else:
            img0 = torch.cat(img0T, 1)
            img1 = torch.cat(img1T, 1)

        # if self.train:
        label = torch.from_numpy(
            np.array([int(img_tuple['label'] != 'PASS')], dtype=np.float32)
        )
        return img0, img1, label

    def __len__(self):
        """Length of Dataset"""
        return len(self.data_frame)

    def get_grid_concat(self, img_list):
        """Grid Concat"""
        x, y = int(self.grid_map["x"]), int(self.grid_map["y"])
        combined_y = []
        cnt = 0
        for _ in range(0, y, 1):
            combined_x = []
            for j in range(0, x, 1):
                combined_x.append(img_list[cnt])
                cnt += 1
                if j == (x - 1):
                    combined_y.append(torch.cat(combined_x, 2))
        img_grid = torch.cat(combined_y, 1)
        return img_grid

    def getTotFail(self):
        """Total FAIL in dataset"""
        return len(self.data_frameFAIL)

    def getTotPass(self):
        """Total PASS in dataset"""
        return len(self.data_framePASS)

    def getComparePathsV1(self, img_tuple):
        """Get compare file Path"""
        return os.path.join(self.input_image_root, img_tuple['input_path'], img_tuple['object_name'])

    def getGoldenPathsV1(self, img_tuple):
        """Get golden file Path"""
        return os.path.join(self.input_image_root, img_tuple['golden_path'], img_tuple['object_name'])


def get_sampler(dataset, train_sampl_ratio=0.1):
    """
    Returns a weighted sampler for imbalanced dataset.

    The weighted sampler increases the sampling rate of FAIL instances relative to PASS instances.

    Args:
        dataset (SiameseNetworkTRIDataset): The input dataset.
        train_sampl_ratio (float): The ratio to increase the sampling rate of FAIL instances.

    Returns:
        WeightedRandomSampler: The weighted random sampler object.
    """
    n = dataset.data_frame.shape[0]
    target_list = [0] * n
    df_ = dataset.data_frame.copy()
    fail_indices = [i for i in range(df_.shape[0]) if df_['label'].iloc[i] != 'PASS']

    for i in fail_indices:
        target_list[i] = 1
    num_pass = dataset.data_frame.shape[0] - len(fail_indices)
    pf_ratio = num_pass / len(fail_indices)
    fail_wt = pf_ratio * train_sampl_ratio

    logging.info('\nSampling Defective components at {:05.2f}:1 rate'.format(fail_wt))
    class_weights = torch.tensor([1, fail_wt], dtype=torch.float)
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True)

    return weighted_sampler
