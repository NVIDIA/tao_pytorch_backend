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

"""File containing vocabulary cookbook for PyTorch models."""

import operator

# Import the required libraries.
from dataclasses import dataclass
from eff.validator.conditions import Expression

from nvidia_tao_pytorch.core.cookbooks.cookbook import Cookbook

__all__ = ["VocabularyCondition", "VocabularyCookbook"]


@dataclass
class VocabularyCondition:
    """Condition used to activate the recipe that restores a vocab from archive.
    Expressions that must be satisfied: EFF format version (=1), dict_archive_format (=1).
    """

    format_version: Expression = Expression(operator.eq, 1)
    vocab_archive_format: Expression = Expression(operator.eq, 1)


class VocabularyCookbook(Cookbook):
    """ Class providing recipes for storing/restoring dicts. """

    @classmethod
    def to_csv(cls, obj: list) -> str:
        """
        Serializes dictionary to "csv" string: one word per line, no commas.

        Args:
            obj: vocabulary (list)

        Returns:
            "Serialized" list.
        """
        return "\n".join(obj)

    @classmethod
    def restore_from(cls, restore_path: str, filename: str) -> dict:
        """Restores and deserializes vocabulary.
        Assumes simple format: one word per line.

        Args:
            archive_path: Path to the archive.
            filename: Name of the file in the archive.

        Returns:
            Dictionary
        """
        # @zeyuz: must process NeMo prefix
        if 'nemo:' in restore_path:
            restore_path = restore_path.split('nemo:')[1]

        # Try to get the content of the file.
        content = cls.get_file_content(archive_path=restore_path, filename=filename)
        # Return vocabulary - list of words.
        return content.split("\n")
