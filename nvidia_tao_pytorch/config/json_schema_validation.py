# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Original source taken from https://github.com/NVIDIA/NeMo
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

"""Utility module to validate json-schema from the api."""

import sys

    
def validate_schema(_value, _properties, hierarchy):
    """
    Recursively validates a JSON schema against a set of properties.

    Args:
        _value (dict): The JSON object to validate.
        _properties (dict): The schema properties to validate against.
        hierarchy (list): A list to keep track of the current hierarchy in the JSON structure.

    Returns:
        str: An error message if the schema is invalid, otherwise None.
    """
    if isinstance(_value, dict):
        for _value_key, _value_obj in _value.items():
            if _value_key == 'automl_default_parameters' or "properties" not in _properties:
                continue

            if _value_key not in _properties["properties"]:
                hierarchy_str = ".".join(hierarchy)
                return f"Invalid schema : key = {_value_key} not present in the {hierarchy_str} config specs. "
            
            hierarchy.append(_value_key)
            status = validate_schema(_value_obj, _properties["properties"][_value_key], hierarchy)
            hierarchy.pop()
            if status:
                return status
            
    hierarchy_str = ".".join(hierarchy)

    if _properties:
        # type check
        if "type" in _properties:
            if _properties["type"] == "integer" and not isinstance(_value, int):
                return f"Type Error : {hierarchy_str} should be of type integer. "
            
            if _properties["type"] == "number" and not (isinstance(_value, float) or isinstance(_value, int)):
                return f"Type Error : {hierarchy_str} should be of type float. "
            
            if _properties["type"] == "boolean" and not isinstance(_value, int):
                return f"Type Error : {hierarchy_str} should be of type boolean. "
            
            if _properties["type"] == "string" and not isinstance(_value, str):
                return f"Type Error : {hierarchy_str} should be of type string. "
            
            # valid_min range check
            if  _properties["type"] == "number" and "minimum" in _properties:
                if float(_value) < float(_properties["minimum"]):
                    return f"Invalid schema : {hierarchy_str} should be >= {str(_properties['minimum'])}, current value is {_value} "

            # valid_max range check
            if  _properties["type"] == "number" and "maximum" in _properties:
                if float(_value) > float(_properties["maximum"]):
                    return f"Invalid schema : {hierarchy_str} should be <= {str(_properties['maximum'])}, current value is '{_value} "

        # valid_options check
        if "enum" in _properties and _value not in _properties["enum"]:
            if 'None' in _properties["enum"] and _value in (None, ''):
                return None
            return f"Invalid schema : Allowed values for the {hierarchy_str} are {_properties['enum']}, current value is \'{_value}\' "

    return None


# json schema validation script
def validate_jsonschema(json_schema, json_metadata):
    """
    Validates a JSON schema against a given metadata schema.

    Args:
        json_schema (dict): The JSON schema to validate.
        json_metadata (dict): The metadata schema to validate against.

    Returns:
        str: An error message if the schema is invalid, otherwise None.
    """
    hierarchy = []
    
    try:
        for key, value_obj in json_schema.items():
            if key == 'automl_default_parameters':
                continue
            
            if key not in json_metadata.keys():
                return f"Invalid schema : key = {key} not present in the config specs."
            
            hierarchy.append(key)
            status = validate_schema(value_obj, json_metadata[key], hierarchy)
            hierarchy.pop()
            if status:
                return status
        return None
    except Exception as err:
        tb = sys.exception().__traceback__
        return f"Invalid schema : {err.with_traceback(tb)}"
