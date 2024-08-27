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

"""Safe Unpickler to avoild unsafe deserialization."""
import pickle
from io import BytesIO


class SafeUnpickler(pickle.Unpickler):
    """
    Custom unpickler that only allows deserialization of a specified class.
    """

    def __init__(self, serialized_data: bytes, class_name: str):
        """
        Initialize the unpickler with the serialized data and the name of the class to allow deserialization for.

        Args:
        serialized_data (bytes): The serialized data to be deserialized.
        class_name (string): The name of the class to be deserialized.
        """
        self.class_name = class_name
        super().__init__(BytesIO(serialized_data))

    def find_class(self, module: str, name: str) -> type:
        """
        Override the default find_class() method to only allow the specified class to be deserialized.

        Args:
        module (string): The module name.
        name (string): The class name.

        Returns:
        type: The specified class.
        """
        # Only allow the specified class to be deserialized
        if name == self.class_name:
            return globals()[name]
        # Raise an exception for all other classes
        raise pickle.UnpicklingError("Invalid class: %s.%s" % (module, name))
