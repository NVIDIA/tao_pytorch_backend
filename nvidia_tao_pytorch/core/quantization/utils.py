# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Utility functions for quantization.

Provides helper routines shared across backends, such as matching modules against user-specified
patterns.
"""

import fnmatch
import torch.nn as nn


def match_layer(module: nn.Module, module_name_in_graph: str, pattern: str) -> bool:
    """Return whether a module matches the given name or pattern.

    The check prioritizes the module's name in the model graph (e.g., ``layers.0.conv1``). If that does
    not match, it falls back to the module's type name (e.g., ``Linear`` or ``Conv2d``). Wildcards are
    supported in ``pattern``.

    Examples
    --------
    >>> import torch.nn as nn
    >>> linear_layer = nn.Linear(10, 20)
    >>> conv_layer = nn.Conv2d(3, 64, 3)
    >>> match_layer(linear_layer, "backbone.classifier.fc", "backbone.classifier.fc")
    True
    >>> match_layer(linear_layer, "backbone.classifier.fc", "Linear")
    True
    >>> match_layer(conv_layer, "backbone.features.conv1", "Linear")
    False
    >>> match_layer(linear_layer, "backbone.classifier.fc", "backbone.classifier.*")
    True
    >>> match_layer(conv_layer, "backbone.features.conv1", "Conv2d")
    True

    Parameters
    ----------
    module : torch.nn.Module
        Module instance to check.
    module_name_in_graph : str
        Module name within the model's graph.
    pattern : str
        Name or pattern (wildcards allowed) to match against.

    Returns
    -------
    bool
        True if the module matches the given name or pattern, False otherwise.

    Raises
    ------
    TypeError
        If any of the arguments are of incorrect type.
    ValueError
        If ``pattern`` is an empty string.

    """
    if module is None:
        raise TypeError("module cannot be None")
    if not isinstance(module_name_in_graph, str):
        raise TypeError(
            f"module_name_in_graph must be a string, but got {type(module_name_in_graph).__name__}"
        )
    if not isinstance(pattern, str):
        raise TypeError(f"pattern must be a string, but got {type(pattern).__name__}")
    if not pattern:
        raise ValueError("pattern cannot be empty")

    if fnmatch.fnmatch(module_name_in_graph, pattern):
        return True

    module_type_name = module.__class__.__name__
    if fnmatch.fnmatch(module_type_name, pattern):
        return True

    return False
