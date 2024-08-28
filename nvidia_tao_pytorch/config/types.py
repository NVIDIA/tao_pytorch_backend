# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""
This module provides utility functions to create dataclass fields with customized metadata for various data types.
Each function in this module is designed to simplify the creation of dataclass fields with predefined metadata,
which can be further customized via keyword arguments. This approach facilitates the definition of models or configurations
where the properties of data fields need to be clearly specified, such as in settings for data validation, serialization,
or user interfaces.

Functions:
    STR_FIELD(value, **meta_args) - Returns a dataclass field for a string with customizable metadata.
    INT_FIELD(value, **meta_args) - Returns a dataclass field for an integer with customizable metadata.
    FLOAT_FIELD(value, **meta_args) - Returns a dataclass field for a float with customizable metadata.
    BOOL_FIELD(value, **meta_args) - Returns a dataclass field for a boolean with customizable metadata.
    LIST_FIELD(arrList, **meta_args) - Returns a dataclass field for a list with customizable metadata.
    DICT_FIELD(hashMap, **meta_args) - Returns a dataclass field for a dictionary with customizable metadata.
    DATACLASS_FIELD(hashMap, **meta_args) - Returns a dataclass field for dataclass instances with customizable metadata.

Each function supports an extensive range of metadata options to define attributes like display name, description, default values,
examples, validation constraints, and dependency relationships among fields.

Usage:
    The module functions can be directly called to create fields in dataclasses, where each field's characteristics
    and behavior are dictated by the provided metadata and the initial value.
"""

from dataclasses import field


def STR_FIELD(value, **meta_args):
    """
    Create a field with string data type, initializing with default settings that can be overridden by kwargs.

    Args:
        value (str): Default value for the field.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value.
    """
    metadata = {
        "display_name": "",
        "value_type": "string",
        "description": "",
        "default_value": "",
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "FALSE",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    if metadata["default_value"] in (None, "") and value not in (None, ""):
        metadata["default_value"] = value
    return field(default=value, metadata=metadata)


def INT_FIELD(value, **meta_args):
    """
    Create a field with integer data type, initializing with default settings that can be overridden by kwargs.

    Args:
        value (int): Default value for the field.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value.
    """
    metadata = {
        "display_name": "",
        "value_type": "int",
        "description": "",
        "default_value": "",
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "FALSE",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    if metadata["default_value"] in (None, "") and value not in (None, ""):
        metadata["default_value"] = value
    return field(default=value, metadata=metadata)


def FLOAT_FIELD(value, **meta_args):
    """
    Create a field with float data type, initializing with default settings that can be overridden by kwargs.

    Args:
        value (float): Default value for the field.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value.
    """
    metadata = {
        "display_name": "",
        "value_type": "float",
        "description": "",
        "default_value": "",
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "FALSE",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    if metadata["default_value"] in (None, "") and value not in (None, ""):
        metadata["default_value"] = value
    return field(default=value, metadata=metadata)


def BOOL_FIELD(value, **meta_args):
    """
    Create a field with boolean data type, initializing with default settings that can be overridden by kwargs.

    Args:
        value (bool): Default value for the field.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value.
    """
    metadata = {
        "display_name": "",
        "value_type": "bool",
        "description": "",
        "default_value": "",
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "FALSE",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    if metadata["default_value"] in (None, "") and value not in (None, ""):
        metadata["default_value"] = value
    return field(default=value, metadata=metadata)


def LIST_FIELD(arrList, **meta_args):
    """
    Create a field for a list, initializing with default settings that can be overridden by kwargs.

    Args:
        arrList (list): Default list to initialize the field with.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value (default factory if specified).
    """
    metadata = {
        "display_name": "",
        "value_type": "list",
        "description": "",
        "default_value": arrList,
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "FALSE",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    return field(default_factory=lambda: arrList, metadata=metadata)


def DICT_FIELD(hashMap, **meta_args):
    """
    Create a field for a dictionary, initializing with default settings that can be overridden by kwargs.

    Args:
        hashMap (dict): Default dictionary to initialize the field with.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value (default factory if specified).
    """
    metadata = {
        "display_name": "",
        "value_type": "collection",
        "description": "",
        "default_value": hashMap,
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "FALSE",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    return field(default_factory=lambda: hashMap, metadata=metadata)


def DATACLASS_FIELD(hashMap, **meta_args):
    """
    Create a field representing a dataclass, initializing with default settings that can be overridden by kwargs.

    Args:
        hashMap (any): Default dataclass instance to initialize the field with.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value (default factory if specified).
    """
    metadata = {
        "display_name": "",
        "value_type": "collection",
        "description": "",
        "default_value": "",
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "FALSE",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    return field(default_factory=lambda: hashMap, metadata=metadata)
