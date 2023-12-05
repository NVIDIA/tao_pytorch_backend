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

import json
import os

if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco")
    ann = os.path.join(dataset_dir, "annotations/lvis_v1_val.json")
    print("Loading", ann)
    data = json.load(open(ann, "r"))
    cat_names = [x["name"] for x in sorted(data["categories"], key=lambda x: x["id"])]
    nonrare_names = [
        x["name"]
        for x in sorted(data["categories"], key=lambda x: x["id"])
        if x["frequency"] != "r"
    ]

    synonyms = [x["synonyms"] for x in sorted(data["categories"], key=lambda x: x["id"])]
    nonrare_synonyms = [
        x["synonyms"]
        for x in sorted(data["categories"], key=lambda x: x["id"])
        if x["frequency"] != "r"
    ]

    with open("datasets/openseg/lvis_1203.txt", "w") as f:
        for idx, cat in enumerate(cat_names):
            cat = cat.replace("_", " ")
            f.write(f"{idx+1}:{cat}\n")

    with open("datasets/openseg/lvis_1203_with_prompt_eng.txt", "w") as f:
        for idx, syns in enumerate(synonyms):
            cat = ",".join(syns)
            cat = cat.replace("_", " ")
            f.write(f"{idx+1}:{cat}\n")

    with open("datasets/openseg/lvis_nonrare_866.txt", "w") as f:
        for idx, cat in enumerate(nonrare_names):
            cat = cat.replace("_", " ")
            f.write(f"{idx+1}:{cat}\n")

    with open("datasets/openseg/lvis_nonrare_866_with_prompt_eng.txt", "w") as f:
        for idx, syns in enumerate(nonrare_synonyms):
            cat = ",".join(syns)
            cat = cat.replace("_", " ")
            f.write(f"{idx+1}:{cat}\n")
