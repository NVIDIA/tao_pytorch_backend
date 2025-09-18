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

"""Validation utilities for the quantization framework."""

from typing import List

from nvidia_tao_pytorch.core.quantization.constants import SupportedDtype


def get_valid_dtype_options() -> List[str]:
    """Return valid dtype options derived from the enum values.

    Returns
    -------
    list[str]
        Valid data type strings (e.g., ["int8", "fp8_e4m3fn", "fp8_e5m2"]).
    """
    return [dtype.value for dtype in SupportedDtype]


def assert_supported_dtype(dtype: str) -> None:
    """Raise a helpful error if dtype is not supported.

    Parameters
    ----------
    dtype : str
        Dtype string to validate.
    """
    if dtype is None:
        raise TypeError("dtype cannot be None")
    valid = get_valid_dtype_options()
    if str(dtype).lower() not in valid:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Supported dtypes: {valid}. "
            "To extend support, add the dtype to SupportedDtype in constants.py."
        )
