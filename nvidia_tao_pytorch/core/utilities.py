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

"""Common utilties for PTL."""

import glob
import os

# Define 1MB for filesize calculation.
MB = 1 << 20


def get_num_trainable_elements(model):
    """Get number of trainable model elements.

    Args:
        model (ptl.module): Pytorch lightning module.

    Return:
        size (int): Number of elements in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_file_size(model_path):
    """Get the size of the model.

    Args:
        model_path (str): UNIX path to the model.

    Returns:
        file_size (float): File size in MB.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file wasn't found at {model_path}")
    file_size = os.path.getsize(model_path) / MB
    return file_size


def update_results_dir(cfg, task):
    """Update global results_dir based on task.results_dir.

    This function should be called at the beginning of a pipeline script.

    Args:
        cfg (Hydra config): Config object loaded by Hydra
        task (str): TAO pipeline name

    Returns:
        Updated cfg
    """
    if cfg[task]['results_dir']:
        cfg.results_dir = cfg[task]['results_dir']
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, task)
        cfg[task]['results_dir'] = cfg.results_dir
    print(f"{task.capitalize()} results will be saved at: {cfg.results_dir}")

    return cfg


def get_last_generated_file(folder_path, extension="txt"):
    """Returns the last generated file in the folder.

    Args:
        folder_path (str): path to the folder
        extension (str): file extension
    """
    files = glob.glob(os.path.join(folder_path, f"*.{extension}"))
    return max(files, key=os.path.getmtime, default=None)
