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
"""Core quantization orchestration utilities for TAO Toolkit.

Exposes the :class:`ModelQuantizer` entry point that coordinates prepare,
optional calibration, quantization, and saving using a pluggable backend.
"""

from __future__ import annotations

from typing import Any, Optional, Type
from nvidia_tao_pytorch.core.tlt_logging import logger

import torch.nn as nn
import torch

from nvidia_tao_pytorch.core.quantization.registry import get_backend_class
from nvidia_tao_pytorch.core.quantization.quantizer_base import QuantizerBase
from nvidia_tao_pytorch.core.quantization.calibratable import Calibratable
import importlib
# Import backends package for side-effect registration of available backends
try:
    importlib.import_module("nvidia_tao_pytorch.core.quantization.backends")
except Exception:
    # Backends are optional; ignore if unavailable at import time
    pass
from nvidia_tao_core.config.common.quantization.default_config import (
    ModelQuantizationConfig,
)
from nvidia_tao_pytorch.core.quantization.backends.modelopt.utils import build_model_quant_config_from_omegaconf
from torch.utils.data import DataLoader


def _coerce_to_model_quant_config(cfg_like: Any) -> ModelQuantizationConfig:
    if isinstance(cfg_like, ModelQuantizationConfig):
        return cfg_like
    # Accept OmegaConf DictConfig or plain dict
    return build_model_quant_config_from_omegaconf(cfg_like)


class ModelQuantizer:
    """High-level interface to run the quantization pipeline.

    This class maintains quantization context (backend, config, etc.) and exposes
    methods to perform the full pipeline: prepare, optionally calibrate, quantize,
    and save.

    Examples
    --------
    Basic usage:
        >>> from nvidia_tao_pytorch.core.quantization import ModelQuantizer
        >>>
        >>> # Create quantizer with config
        >>> config = {"backend": "modelopt", "layers": [...]}
        >>> quantizer = ModelQuantizer(config)
        >>>
        >>> # Full pipeline
        >>> quantized_model = quantizer.quantize_model(model)
        >>> quantizer.save_model(path="quantized_model.pth")

    Step-by-step usage with calibration:
        >>> # Prepare model
        >>> prepared_model = quantizer.prepare(model)
        >>>
        >>> # Calibrate (if backend supports it)
        >>> quantizer.calibrate(prepared_model, calibration_dataloader)
        >>>
        >>> # Quantize
        >>> quantized_model = quantizer.quantize()
        >>>
        >>> # Save
        >>> quantizer.save_model(path="quantized_model.pth")
    """

    def __init__(self, cfg_like: Any):
        """Initialize the quantizer with configuration.

        Parameters
        ----------
        cfg_like : Any
            Configuration for quantization. Accepts a ``ModelQuantizationConfig``,
            an OmegaConf ``DictConfig``, or a plain ``dict``.
        """
        self.config = _coerce_to_model_quant_config(cfg_like)

        if not self.config.backend:
            raise ValueError("Quantization backend must be specified in the config (e.g., 'modelopt').")

        self.backend_class: Type[QuantizerBase] = get_backend_class(self.config.backend)
        if not issubclass(self.backend_class, QuantizerBase):
            raise TypeError(
                f"Backend '{self.config.backend}' must be a subclass of QuantizerBase, "
                f"but got {self.backend_class}"
            )
        self.quantizer: QuantizerBase = self.backend_class()
        self.prepared_model: Optional[nn.Module] = None
        self.quantized_model: Optional[nn.Module] = None

        logger.info(f"Quantization backend selected: {self.config.backend}")

    def prepare(self, model: nn.Module) -> nn.Module:
        """Prepare the model for quantization.

        Parameters
        ----------
        model : torch.nn.Module
            Model to prepare for quantization.

        Returns
        -------
        torch.nn.Module
            Prepared model (e.g., after observer/fake-quant insertion depending on backend).
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")

        logger.debug("Preparing model for quantization...")
        self.prepared_model = self.quantizer.prepare(model, self.config)
        logger.info("Model preparation complete")
        return self.prepared_model

    def calibrate(self, model: nn.Module, dataloader: DataLoader) -> None:
        """Calibrate a prepared model using the provided data loader.

        Parameters
        ----------
        model : torch.nn.Module
            Prepared model to calibrate.
        dataloader : torch.utils.data.DataLoader
            Data loader providing calibration samples.

        Notes
        -----
        If the selected backend does not support calibration, this call is a no-op with a warning.
        """
        if isinstance(self.quantizer, Calibratable):
            logger.info("Starting calibration phase")
            self.quantizer.calibrate(model, dataloader)
            logger.info("Calibration setup complete")
        else:
            logger.warning(
                f"Backend '{self.config.backend}' does not support calibration; skipping calibration phase"
            )

    def quantize(self, model: Optional[nn.Module] = None) -> nn.Module:
        """Quantize the prepared model.

        Parameters
        ----------
        model : torch.nn.Module, optional
            Prepared model to quantize. If ``None``, the most recently prepared model is used.

        Returns
        -------
        torch.nn.Module
            Quantized model.
        """
        if model is None:
            if self.prepared_model is None:
                raise ValueError("No model has been prepared. Call prepare() first.")
            model = self.prepared_model

        logger.info("Quantizing model...")
        self.quantized_model = self.quantizer.quantize(model, self.config)
        logger.info("Quantization complete")
        return self.quantized_model

    def save_model(self, model: Optional[nn.Module] = None, path: str = "") -> None:
        """Save a quantized model to disk.

        Parameters
        ----------
        model : torch.nn.Module, optional
            Model instance to save. If ``None``, the most recently quantized model is used.
        path : str
            Directory path where the model should be saved.

        Returns
        -------
        None
        """
        if model is None:
            if self.quantized_model is None:
                raise ValueError("No model has been quantized. Call quantize() first.")
            model = self.quantized_model

        if hasattr(self.quantizer, 'save_model'):
            logger.info(f"Saving quantized model to directory: {path}")
            self.quantizer.save_model(model, path)
        else:
            # Fallback to torch.save if backend doesn't provide save_model
            logger.info(f"Saving quantized model state_dict to path: {path}")
            torch.save(model.state_dict(), path)

    def quantize_model(self, model: nn.Module, calibration_loader: Optional[DataLoader] = None) -> nn.Module:
        """Run the end-to-end quantization pipeline and return the quantized model.

        Parameters
        ----------
        model : torch.nn.Module
            Model to quantize.
        calibration_loader : torch.utils.data.DataLoader, optional
            Calibration data loader. If provided and the backend supports calibration,
            calibration is performed when the config mode is "static_ptq".

        Returns
        -------
        torch.nn.Module
            Quantized model.
        """
        self.prepare(model)
        if calibration_loader is not None and isinstance(self.quantizer, Calibratable) and self.config.mode == "static_ptq":
            self.calibrate(model, calibration_loader)
        return self.quantize()
