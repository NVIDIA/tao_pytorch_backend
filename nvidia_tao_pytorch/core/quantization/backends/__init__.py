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

"""Quantization backends for TAO Toolkit."""

from nvidia_tao_pytorch.core.quantization.registry import (
    register_backend,
    get_available_backends,
    get_backend_class,
)

# Import concrete backends so they self-register via decorators.
# These imports are intentionally placed here to provide a seamless user
# experience – importing the quantization package will make backends
# available in the registry.
try:  # pragma: no cover - import side-effect only
    from nvidia_tao_pytorch.core.quantization.backends.modelopt import ModelOptBackend  # noqa: F401
except Exception:
    # ModelOpt is optional; ignore if unavailable at import time
    pass

# Import TorchAO backend so it self-registers via decorator
try:  # pragma: no cover - import side-effect only
    from .torchao.torchao import TorchAOBackend  # noqa: F401
except Exception:
    # TorchAO is optional; ignore if unavailable at import time
    pass

__all__ = [
    "register_backend",
    "get_available_backends",
    "get_backend_class",
]
