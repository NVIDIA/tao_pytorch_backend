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

"""TAO common path utils used across all apps."""

import os


def expand_path(path):
    """Function to resolve a path.

    This function takes in a path and returns the absolute path of that path after
    expanding the tilde (~) character to the user's home directory to prevent path
    traversal vulnerability.

    Args:
        path (str): The path to expand and make absolute.

    Returns:
        str: The absolute path with expanded tilde.
    """
    return os.path.abspath(os.path.expanduser(path))
