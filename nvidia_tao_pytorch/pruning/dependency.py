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

""" Prune modules as Monkey patch for torch-pruning. """

from torch_pruning import DependencyGraph
from torch_pruning import ops


class TAO_DependencyGraph(DependencyGraph):
    """Inherit DependencyGraph class from torch-pruning"""

    def update_index_mapping(self):
        """ Update all index mapping after pruning """
        for _, node in self.module2node.items():
            if node.type == ops.OPTYPE.CONCAT:
                # enable index mapping for the concat in FPN neck
                for node_out in node.outputs:
                    if "linear_fuse.conv" in node_out.name:
                        node.enable_index_mapping = True
                self._update_concat_index_mapping(node)
            if node.type == ops.OPTYPE.SPLIT:
                self._update_split_index_mapping(node)
            if node.type == ops.OPTYPE.RESHAPE:
                self._update_reshape_index_mapping(node)
