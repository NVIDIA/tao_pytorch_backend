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

"""Utils for metric-learning recognition."""

import os


def no_folders_in(path_to_parent):
    """Checks whether folders exist in the directory.

    Args:
        path_to_parent (String): a directory for an image file or folder.

    Returns:
        no_folders (Boolean): If true, the directory is an image folder, otherwise it's a classifcation folder.
    """
    no_folders = True
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent, fname)):
            no_folders = False
            break

    return no_folders
