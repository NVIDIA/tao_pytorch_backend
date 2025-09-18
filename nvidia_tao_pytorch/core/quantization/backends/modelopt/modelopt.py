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

"""ModelOpt quantization backend for TAO Toolkit.

Implements a backend that translates TAO quantization configuration to the
ModelOpt configuration dict and invokes ``modelopt.torch.quantization``
APIs for calibration and quantization.
"""

from __future__ import annotations

from typing import Callable, Optional
from nvidia_tao_pytorch.core.tlt_logging import logger as tlt_logger

import torch
import torch.nn as nn

from nvidia_tao_pytorch.core.quantization.quantizer_base import QuantizerBase
from nvidia_tao_pytorch.core.quantization.calibratable import Calibratable
from nvidia_tao_pytorch.core.quantization.registry import register_backend
from nvidia_tao_pytorch.core.quantization.constants import QuantizationMode
from nvidia_tao_pytorch.core.quantization.backends.modelopt.utils import convert_tao_to_modelopt_config

from nvidia_tao_core.config.common.quantization.default_config import ModelQuantizationConfig
from tqdm import tqdm
import os

try:
    import modelopt.torch.quantization as mtq  # type: ignore
    import modelopt.torch.opt as mto

except Exception as exc:  # pragma: no cover - import error path
    raise ImportError(
        "modelopt is not installed or failed to import. Install ModelOpt (pip install nvidia-modelopt) "
        "to use the 'modelopt' backend."
    ) from exc


SUPPORTED_MODES = {QuantizationMode.STATIC_PTQ.name.lower()}


def _default_forward_loop(model: nn.Module) -> None:
    """No-op forward loop used when no calibration data is provided.

    Parameters
    ----------
    model : torch.nn.Module
        Model to run through a dummy evaluation pass.
    """
    model.eval()
    with torch.no_grad():
        return


@register_backend("modelopt")
class ModelOptBackend(QuantizerBase, Calibratable):
    """ModelOpt quantization backend.

    Adapts the TAO quantization configuration to the ModelOpt configuration
    dictionary and delegates quantizer insertion and calibration to
    ``modelopt.torch.quantization`` APIs.
    """

    def __init__(self) -> None:
        self._forward_loop: Optional[Callable[[nn.Module], None]] = None
        self.backend_name = "modelopt"  # Store the backend name as an instance attribute
        self._logger = tlt_logger

    def prepare(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """Validate inputs and return the model unchanged.

        ModelOpt handles quantizer insertion inside its ``quantize`` API, so
        ``prepare`` is effectively a no-op for this backend.

        Parameters
        ----------
        model : torch.nn.Module
            Model to prepare.
        config : ModelQuantizationConfig
            Quantization configuration.

        Returns
        -------
        torch.nn.Module
            The input model unchanged.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        if not isinstance(config, ModelQuantizationConfig):
            raise TypeError("config must be an instance of ModelQuantizationConfig")
        # Accept both enum and string for mode; normalize to a lower-case string
        mode_value = (
            config.mode.name.lower() if isinstance(config.mode, QuantizationMode) else str(config.mode).lower()
        )
        if mode_value not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode '{config.mode}' for backend '{self.backend_name}'. "
                f"Supported modes: {sorted(SUPPORTED_MODES)}"
            )
        self._logger.debug("ModelOptBackend.prepare: input validation complete; returning model unchanged")
        return model

    def calibrate(self, model: nn.Module, data_loader) -> None:
        """Build and store a forward loop for later calibration.

        Parameters
        ----------
        model : torch.nn.Module
            Model to be calibrated.
        data_loader : torch.utils.data.DataLoader
            Iterator producing inputs. Each batch can be a tensor, a tuple where the first
            element is the tensor, or a dict with key "input".
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")

        def _extract_input(batch):
            if torch.is_tensor(batch):
                return batch
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                return batch[0]
            if isinstance(batch, dict):
                # Common convention
                return batch.get("input", next(iter(batch.values())))
            raise TypeError(
                f"Unsupported batch format for calibration. Expected a Tensor, (Tensor, ...), or dict,"
                f" but got type {type(batch).__name__}."
            )

        def forward_loop(m: nn.Module) -> None:
            if len(data_loader) == 0:
                self._logger.warning(
                    "No calibration data present in the calibration dataset. "
                    "Accuracy of the quantized model may degrade if activations are being quantized. "
                    "Please check the calibration dataset `quant_calibration_dataset` if this is a problem. "
                    "Weight only quantization will not be affected. "
                    "Disregard this warning if you are running evaluation, inference, or other non-quantization tasks."
                )
            m.eval().cuda()
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Calibrating model"):
                    x = _extract_input(batch).cuda()
                    m(x)
                    # break
        self._forward_loop = forward_loop
        self._logger.debug("Calibration forward loop has been set")

    def quantize(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """Quantize a model using ModelOpt APIs.

        Translates the TAO configuration into a ModelOpt configuration dict and
        invokes ``modelopt.torch.quantization.quantize`` with the stored forward
        loop if available.

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
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        if not isinstance(config, ModelQuantizationConfig):
            raise TypeError("config must be an instance of ModelQuantizationConfig")

        modelopt_cfg = convert_tao_to_modelopt_config(config, model)

        # ModelOpt configuration prepared; proceed to quantization

        if self._forward_loop is None:
            self._logger.warning(
                "No calibration dataloader provided; using a no-op forward loop. "
                "Accuracy of the quantized model may degrade if activations are being quantized"
                "and no calibration data is provided. Weight only quantization will not be affected. "
                "Disregard this warning if you are running evaluation, inference, or other non-quantization tasks."
            )
            forward_loop = _default_forward_loop
        else:
            forward_loop = self._forward_loop

        self._logger.info("Invoking ModelOpt quantization")
        quantized_model = mtq.quantize(model, modelopt_cfg, forward_loop)
        self._logger.info("ModelOpt quantization complete")
        return quantized_model

    def save_model(self, model: nn.Module, path: str) -> None:
        """Save the quantized model to a directory.

        Parameters
        ----------
        model : torch.nn.Module
            Quantized model to save.
        path : str
            Directory where the model is saved as ``quantized_model_modelopt.pth``.
        """
        # Save the quantized model as a Lightning checkpoint of the original module

        quantized_model_path = os.path.join(path, f"quantized_model_{self.backend_name}.pth")
        mto.save(model, quantized_model_path)
        self._logger.info(f"Quantized model saved to: {quantized_model_path}")


__all__ = [
    "ModelOptBackend",
]
