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

"""Constants for the quantization framework."""

from enum import Enum, auto


class SupportedDtype(Enum):
    """Supported data types for quantization."""

    INT8 = "int8"
    FP8_E4M3FN = "fp8_e4m3fn"
    FP8_E5M2 = "fp8_e5m2"


class QuantizationMode(Enum):
    """Supported quantization modes."""

    WEIGHT_ONLY_PTQ = auto()
    STATIC_PTQ = auto()


# Registry dictionaries for backward compatibility
BACKEND_REGISTRY = {}
OBSERVER_REGISTRY = {}
FAKE_QUANT_REGISTRY = {}
