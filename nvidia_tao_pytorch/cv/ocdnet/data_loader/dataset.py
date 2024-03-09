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
"""Dataset module."""
import pathlib
import cv2
import numpy as np

from nvidia_tao_pytorch.cv.ocdnet.base.base_dataset import BaseDataSet
from nvidia_tao_pytorch.cv.ocdnet.utils import order_points_clockwise, get_datalist, get_datalist_uber

import multiprocessing


class UberDataset(BaseDataSet):
    """Uber Dataset class."""

    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        """Initialize."""
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """Load data."""
        pool = multiprocessing.Pool(processes=4)
        data_list = pool.apply_async(get_datalist_uber, args=(data_path,)).get()
        pool.close()
        pool.join()

        t_datalist = []
        pool = multiprocessing.Pool(processes=4)
        for img_path, label_path in data_list:
            tmp = pool.apply_async(self._get_annotation, args=(label_path,))
            data = tmp.get()
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_datalist.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        pool.close()
        pool.join()

        return t_datalist

    def _get_annotation(self, label_path: str) -> dict:
        polys = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f:
                content = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split('\t')
                params = content[0].split(" ")[:-2]
                try:
                    poly = np.array(list(map(float, params))).reshape(-1, 2).astype(np.float32)
                    if cv2.contourArea(poly) > 0:
                        polys.append(poly)
                        label = content[1]
                        if len(label.split(" ")) > 1:
                            label = "###"
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except Exception:
                    print('load label failed on {}'.format(label_path))
        data = {
            'text_polys': np.array(polys),
            'texts': texts,
            'ignore_tags': ignores,
        }

        return data


class ICDAR2015Dataset(BaseDataSet):
    """ICDAR2015 Dataset."""

    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        """Initialize."""
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """Load data."""
        data_list = get_datalist(data_path)
        t_datalist = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_datalist.append(item)
            else:
                print(f'there is no suit bbox in {label_path}')

        return t_datalist

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if not (box > -50).all():
                        continue
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = ','.join(params[8:])
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except Exception:
                    print(f'load label failed on {label_path}')
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }

        return data
