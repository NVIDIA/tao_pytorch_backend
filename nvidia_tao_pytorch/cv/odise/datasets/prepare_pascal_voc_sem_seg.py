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
from pathlib import Path
import shutil

import numpy as np
import tqdm
from PIL import Image


def convert(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    # do nothing
    Image.fromarray(img).save(output)


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "pascal_voc_d2"
    voc_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "VOCdevkit/VOC2012"
    for split in ["training", "validation"]:
        if split == "training":
            img_name_path = voc_dir / "ImageSets/Segmentation/train.txt"
        else:
            img_name_path = voc_dir / "ImageSets/Segmentation/val.txt"
        img_dir = voc_dir / "JPEGImages"
        ann_dir = voc_dir / "SegmentationClass"

        output_img_dir = dataset_dir / "images" / split
        output_ann_dir = dataset_dir / "annotations_pascal21" / split

        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_ann_dir.mkdir(parents=True, exist_ok=True)

        with open(img_name_path) as f:
            for line in tqdm.tqdm(f.readlines()):
                img_name = line.strip()
                img_path = img_dir / f"{img_name}.jpg"
                ann_path = ann_dir / f"{img_name}.png"

                # print(f'copy2 {output_img_dir}')
                shutil.copy2(img_path, output_img_dir)
                # print(f"convert {ann_dir} to {output_ann_dir / f'{img_name}.png'}")
                convert(ann_path, output_ann_dir / f"{img_name}.png")
