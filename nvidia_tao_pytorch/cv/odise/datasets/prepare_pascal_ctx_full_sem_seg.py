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

import os
import numpy as np
from pathlib import Path
from PIL import Image
import scipy.io as sio

import tqdm


def generate_labels(mat_file, out_dir):

    mat = sio.loadmat(mat_file)
    label_map = mat["LabelMap"]
    assert label_map.dtype == np.uint16
    label_map[label_map == 0] = 65535
    label_map = label_map - 1
    label_map[label_map == 65534] = 65535

    out_file = out_dir / Path(mat_file.name).with_suffix(".tif")
    Image.fromarray(label_map).save(out_file)


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "pascal_ctx_d2"
    voc_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "VOCdevkit/VOC2010"
    mat_dir = voc_dir / "trainval"
    for split in ["training", "validation"]:
        file_names = list((dataset_dir / "images" / split).glob("*.jpg"))
        output_img_dir = dataset_dir / "images" / split
        output_ann_dir = dataset_dir / "annotations_ctx459" / split

        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_ann_dir.mkdir(parents=True, exist_ok=True)

        for file_name in tqdm.tqdm(file_names):
            mat_file_path = mat_dir / f"{file_name.stem}.mat"

            generate_labels(mat_file_path, output_ann_dir)
