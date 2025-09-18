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

"""Quantization core for TAO Toolkit."""

# Core abstract classes
from nvidia_tao_pytorch.core.quantization.quantizer_base import QuantizerBase
from nvidia_tao_pytorch.core.quantization.calibratable import Calibratable

# Configuration classes
from nvidia_tao_core.config.common.quantization.default_config import (
    ModelQuantizationConfig,
    LayerQuantizationConfig,
    WeightQuantizationConfig,
    ActivationQuantizationConfig,
    BaseQuantizationConfig,
)

# Constants and enums
from nvidia_tao_pytorch.core.quantization.constants import (
    QuantizationMode,
)

# Validation utilities
from nvidia_tao_pytorch.core.quantization.validation import get_valid_dtype_options

# Registry management
from nvidia_tao_pytorch.core.quantization.registry import (
    register_observer,
    register_fake_quant,
    register_backend,
    get_available_backends,
    get_backend_class,
    get_available_observers,
    get_available_fake_quants,
    get_observer_class,
    get_fake_quant_class,
    get_registry_manager,
)

# Quantization main
from nvidia_tao_pytorch.core.quantization.quantizer import ModelQuantizer

__all__ = [
    # Core abstract classes
    "QuantizerBase",
    "Calibratable",
    # Configuration classes
    "ModelQuantizationConfig",
    "LayerQuantizationConfig",
    "WeightQuantizationConfig",
    "ActivationQuantizationConfig",
    "BaseQuantizationConfig",
    # Constants and enums
    "QuantizationMode",
    # Validation utilities
    "get_valid_dtype_options",
    # Registry management
    "register_observer",
    "register_fake_quant",
    "register_backend",
    "get_available_backends",
    "get_backend_class",
    "get_available_observers",
    "get_available_fake_quants",
    "get_observer_class",
    "get_fake_quant_class",
    "get_registry_manager",
    # Quantization main
    "ModelQuantizer",
]
