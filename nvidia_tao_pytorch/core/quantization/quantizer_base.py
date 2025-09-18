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

"""Quantizer core for TAO Toolkit."""

from abc import ABC, abstractmethod
import torch.nn as nn

from nvidia_tao_core.config.common.quantization.default_config import (
    ModelQuantizationConfig,
)


class QuantizerBase(ABC):
    """Abstract interface for quantization backends.

    Subclasses implement the backend-specific logic for inserting observers/fake
    quantizers and converting a model to its quantized form.

    See Also
    --------
    Calibratable
        Mix-in interface adding ``calibrate`` for PTQ backends.
    """

    @abstractmethod
    def prepare(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """Insert observers/fake quantizers based on configuration.

        Parameters
        ----------
        model : torch.nn.Module
            Model to prepare for quantization.
        config : ModelQuantizationConfig
            Quantization configuration.

        Returns
        -------
        torch.nn.Module
            Prepared model with observers/fake-quant modules inserted.

        Raises
        ------
        NotImplementedError
            Always, unless implemented by a subclass.
        """
        raise NotImplementedError("Calling abstract method - prepare. Subclass must implement this method.")

    @abstractmethod
    def quantize(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """Convert a prepared model to its quantized form.

        Parameters
        ----------
        model : torch.nn.Module
            Prepared model to quantize.
        config : ModelQuantizationConfig
            Quantization configuration.

        Returns
        -------
        torch.nn.Module
            Quantized model.

        Raises
        ------
        NotImplementedError
            Always, unless implemented by a subclass.
        """
        raise NotImplementedError("Calling abstract method - quantize. Subclass must implement this method.")
