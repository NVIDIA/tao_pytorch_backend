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

"""Utility functions for the ModelOpt backend.

This module provides helpers to translate TAO Toolkit quantization
configuration dataclasses into a configuration dictionary that is
accepted by the Model Optimizer (``modelopt``) PyTorch quantization
APIs.

At the moment the conversion focuses on the most common use-cases –
INT8/PTQ and FP8/Fake-quant flows – and intentionally keeps the
resulting structure simple. It can be extended later as the higher
level TAO configuration gains more features or as we add support for
additional calibration algorithms.

Notes
-----
modelopt represents the *quantization specification* as a dict with two
entries:
    1. ``"quant_cfg"`` – mapping from module/quantizer name patterns to
       ``QuantizerAttributeConfig`` dictionaries (or lists of them).
    2. ``"algorithm"`` – calibration / optimisation algorithm (e.g.
       "max", "smoothquant", "awq_lite" …).

The TAO classes group the same information slightly differently. The
:class:`~nvidia_tao_pytorch.core.quantization.ModelQuantizationConfig`
contains a list of
:class:`~nvidia_tao_pytorch.core.quantization.LayerQuantizationConfig`
objects, each of which can specify weight and / or activation
quantisation parameters.

``convert_tao_to_modelopt_config`` walks over those objects and emits a
flat ``modelopt`` configuration that preserves the intent of the
original specification.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from nvidia_tao_pytorch.core.tlt_logging import logger as tlt_logger

from nvidia_tao_pytorch.core.quantization.utils import match_layer

from nvidia_tao_pytorch.core.quantization import (
    ModelQuantizationConfig,
    LayerQuantizationConfig,
    WeightQuantizationConfig,
    ActivationQuantizationConfig,
)
from nvidia_tao_pytorch.core.quantization.constants import QuantizationMode
from nvidia_tao_pytorch.core.quantization.validation import assert_supported_dtype

__all__ = [
    "convert_tao_to_modelopt_config",
    "build_model_quant_config_from_omegaconf",
]


def _dtype_to_num_bits(dtype: str, mode: str | None = None) -> int | Tuple[int, int]:
    """Translate TAO ``dtype`` strings to ModelOpt ``num_bits`` values.

    Parameters
    ----------
    dtype : str
        Dtype string from TAO configuration (e.g., "int8", "fp8_e4m3fn", "fp8_e5m2"). The mapping
        is not exhaustive; extend as the TAO side gains support for more formats.
    mode : str | None, optional
        Quantization mode (e.g., "static_ptq"). Used to determine if fallback is needed for e5m2.

    Returns
    -------
    int | tuple[int, int]
        Value suitable for the ``num_bits`` field of a ModelOpt ``QuantizerAttributeConfig``.
    """
    _mapping: dict[str, int | Tuple[int, int]] = {
        # Integer formats
        "int8": 8,
        # Floating-point emulations
        # Only the formats enumerated in SupportedDtype are accepted
        "fp8_e4m3fn": (4, 3),
        "fp8_e5m2": (5, 2),
    }

    key = dtype.lower()
    if key not in _mapping:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Supported dtypes: {list(_mapping.keys())}."
        )

    # Check if e5m2 is being used with static mode and fall back to e4m3
    if key == "fp8_e5m2" and mode and "static" in mode.lower():
        tlt_logger.warning(
            f"Static mode is unsupported for float8 e5m2 dtype. Falling back to e4m3. "
            f"Original dtype: {dtype}, mode: {mode}"
        )
        return _mapping["fp8_e4m3fn"]

    return _mapping[key]


def _build_quantizer_cfg(
    qcfg: WeightQuantizationConfig | ActivationQuantizationConfig,
    mode: str | None = None,
) -> Dict[str, Any]:
    """Convert TAO weight/activation config to a ModelOpt quantizer dictionary.

    Parameters
    ----------
    qcfg : WeightQuantizationConfig | ActivationQuantizationConfig
        TAO configuration for weights or activations.
    mode : str | None, optional
        Quantization mode (e.g., "static_ptq"). Used to determine if fallback is needed for e5m2.

    Returns
    -------
    dict
        ModelOpt quantizer attributes dictionary.
    """
    # Special-case: allow explicit "native" to disable a specific quantizer
    # instead of erroring. This enables per-quantizer opt-out without relying on
    # skip patterns.
    dtype_value = str(getattr(qcfg, "dtype", "")).lower()
    if dtype_value == "native":
        return {"enable": False}

    # Validate dtype against SupportedDtype for helpful error messages
    assert_supported_dtype(qcfg.dtype)

    cfg: Dict[str, Any] = {
        "num_bits": _dtype_to_num_bits(qcfg.dtype, mode),
    }
    if qcfg.quant_axis is not None:
        cfg["axis"] = qcfg.quant_axis
    return cfg


def _normalize_dtype(dtype: str) -> str:
    """Normalize dtype strings from YAML/OmegaConf.

    Converts the prefix "float8_" to "fp8_" to match internal expectations.

    Parameters
    ----------
    dtype : str
        Input dtype string.

    Returns
    -------
    str
        Normalized dtype string.
    """
    return dtype.replace("float8_", "fp8_") if isinstance(dtype, str) else dtype


def build_model_quant_config_from_omegaconf(qcfg: DictConfig | dict) -> ModelQuantizationConfig:
    """Construct a ``ModelQuantizationConfig`` from an OmegaConf config or dict.

    Normalizes dtype strings (e.g., "float8_*" -> "fp8_*") and builds the corresponding TAO
    dataclasses.

    Parameters
    ----------
    qcfg : omegaconf.DictConfig | dict
        User-provided configuration.

    Returns
    -------
    ModelQuantizationConfig
        Normalized TAO quantization configuration object.
    """
    cfg_dict = (
        OmegaConf.to_container(qcfg, resolve=True)
        if hasattr(qcfg, "_get_full_key")
        else dict(qcfg)
    )

    layers: list[LayerQuantizationConfig] = []
    for layer in cfg_dict.get("layers", []) or []:
        weights_cfg = None
        if isinstance(layer.get("weights"), dict):
            w = dict(layer["weights"])  # shallow copy
            if "dtype" in w:
                w["dtype"] = _normalize_dtype(w["dtype"])
            weights_cfg = WeightQuantizationConfig(**w)

        activations_cfg = None
        if isinstance(layer.get("activations"), dict):
            a = dict(layer["activations"])  # shallow copy
            if "dtype" in a:
                a["dtype"] = _normalize_dtype(a["dtype"])
            activations_cfg = ActivationQuantizationConfig(**a)

        layers.append(
            LayerQuantizationConfig(
                module_name=layer["module_name"],
                weights=weights_cfg,
                activations=activations_cfg,
            )
        )

    # Normalize/validate mode: accept enum name or string
    mode_value = cfg_dict.get("mode", "static_ptq")
    if isinstance(mode_value, QuantizationMode):
        normalized_mode = mode_value.name.lower()
    else:
        normalized_mode = str(mode_value).lower()

    # Normalise algorithm, if provided. Keep None if unspecified to allow defaults downstream.
    algorithm_value = cfg_dict.get("algorithm", None)
    normalized_algorithm = None
    if algorithm_value is not None:
        normalized_algorithm = str(algorithm_value).lower()

    return ModelQuantizationConfig(
        backend=cfg_dict.get("backend", "modelopt"),
        layers=layers,
        skip_names=cfg_dict.get("skip_names", []) or [],
        mode=normalized_mode,
        algorithm=normalized_algorithm,
    )


def convert_tao_to_modelopt_config(
    config: ModelQuantizationConfig,
    model: nn.Module | None = None,
) -> Dict[str, Any]:
    """Convert TAO configuration into a ModelOpt configuration dict.

    The result strictly follows the ModelOpt schema with top-level keys "quant_cfg" and "algorithm".
    TAO layer patterns are expanded into ModelOpt-compliant keys.

    Parameters
    ----------
    config : ModelQuantizationConfig
        TAO quantization configuration.
    model : torch.nn.Module, optional
        If provided, module name patterns are expanded to qualified names; otherwise, class keys or
        wildcard quantizer names are emitted.

    Returns
    -------
    dict
        ModelOpt configuration dictionary with keys ``quant_cfg`` and ``algorithm``.
    """
    if config is None:
        raise TypeError("config cannot be None")

    # Warn if default dtypes are set to non-native values; feature not supported yet
    default_layer_dtype = str(getattr(config, "default_layer_dtype", "native")).lower()
    default_activation_dtype = str(getattr(config, "default_activation_dtype", "native")).lower()
    if default_layer_dtype != "native" or default_activation_dtype != "native":
        tlt_logger.warning(
            "Non-native default_layer_dtype/default_activation_dtype is currently not supported "
            "by the modelopt backend and will be ignored."
        )

    quant_cfg: Dict[str, Any] = {}

    def _nn_class_key(pattern: str) -> str | None:
        """Return ModelOpt class key (e.g., "nn.Linear") if pattern is a torch.nn class name.

        If the pattern already looks like a class key (starts with "nn."), return as-is.
        """
        if not isinstance(pattern, str) or not pattern:
            return None
        if pattern.startswith("nn."):
            return pattern
        # Best-effort: map capitalized names from torch.nn to "nn.<Class>"
        cls = getattr(nn, pattern, None)
        if isinstance(cls, type) and issubclass(cls, nn.Module):
            return f"nn.{pattern}"
        return None

    def _add_module_rules(key: str, w_cfg: Dict[str, Any] | None, a_cfg: Dict[str, Any] | None) -> None:
        """Insert rules for a module-name pattern by creating quantizer-name keys."""
        if w_cfg is not None:
            quant_cfg[f"{key}.weight_quantizer"] = w_cfg
        if a_cfg is not None:
            quant_cfg[f"{key}.input_quantizer"] = a_cfg

    def _add_class_rules(class_key: str, w_cfg: Dict[str, Any] | None, a_cfg: Dict[str, Any] | None) -> None:
        """Insert rules under a ModelOpt class key (e.g., "nn.Linear")."""
        entry = quant_cfg.setdefault(class_key, {})
        if w_cfg is not None:
            entry["*weight_quantizer"] = w_cfg
        if a_cfg is not None:
            entry["*input_quantizer"] = a_cfg

    # Collect modules and their qualified names if a model was provided.
    named_modules: list[tuple[str, nn.Module]] = []
    if model is not None:
        # ``named_modules`` returns the top-level module with an empty string name – skip that.
        named_modules = [(n, m) for n, m in model.named_modules() if n]

    # First pass – layer-wise specifications
    for layer in config.layers or []:
        if not isinstance(layer, LayerQuantizationConfig):
            raise TypeError(
                "Expected items of 'layers' to be LayerQuantizationConfig instances, "
                f"but got {type(layer)}."
            )

        if not layer.module_name:
            raise ValueError("module_name in LayerQuantizationConfig cannot be empty")

        patterns_matched = False

        if named_modules:
            for qual_name, module in named_modules:
                if match_layer(module, qual_name, layer.module_name):
                    patterns_matched = True
                    _add_module_rules(
                        qual_name,
                        _build_quantizer_cfg(layer.weights, config.mode) if layer.weights is not None else None,
                        _build_quantizer_cfg(layer.activations, config.mode) if layer.activations is not None else None,
                    )
        # If we did not find a match (e.g., model not provided or pattern unmatched),
        # translate the original pattern into ModelOpt-compliant keys.
        if not patterns_matched:
            class_key = _nn_class_key(layer.module_name)
            if class_key is not None:
                _add_class_rules(
                    class_key,
                    _build_quantizer_cfg(layer.weights, config.mode) if layer.weights is not None else None,
                    _build_quantizer_cfg(layer.activations, config.mode) if layer.activations is not None else None,
                )
            else:
                _add_module_rules(
                    layer.module_name,
                    _build_quantizer_cfg(layer.weights, config.mode) if layer.weights is not None else None,
                    _build_quantizer_cfg(layer.activations, config.mode) if layer.activations is not None else None,
                )

    # Second pass – skip patterns override previously defined rules
    for skip_pattern in config.skip_names or []:
        patterns_matched = False
        if named_modules:
            for qual_name, module in named_modules:
                if match_layer(module, qual_name, skip_pattern):
                    patterns_matched = True
                    # Disable both weight and input quantizers for the matched module
                    quant_cfg[f"{qual_name}.weight_quantizer"] = {"enable": False}
                    quant_cfg[f"{qual_name}.input_quantizer"] = {"enable": False}
        if not patterns_matched:
            class_key = _nn_class_key(skip_pattern)
            if class_key is not None:
                entry = quant_cfg.setdefault(class_key, {})
                entry["*weight_quantizer"] = {"enable": False}
                entry["*input_quantizer"] = {"enable": False}
            else:
                quant_cfg[f"{skip_pattern}.weight_quantizer"] = {"enable": False}
                quant_cfg[f"{skip_pattern}.input_quantizer"] = {"enable": False}

    # Safeguard – if user didn't specify anything we disable quantization by default.
    if not quant_cfg:
        quant_cfg["default"] = {"enable": False}

    # Determine calibration/optimisation algorithm from network-level config.
    # Priority: explicit config.algorithm, otherwise default based on config.mode.
    algorithm: str | None
    if getattr(config, "algorithm", None):
        algorithm = str(config.algorithm).lower()
    else:
        mode_value = getattr(config, "mode", None)
        mode_str = str(mode_value).lower() if mode_value is not None else "static_ptq"
        # For PTQ variants, default to "minmax" (matches TAO schema/tests). For non-PTQ, set None.
        if mode_str in {"static_ptq", "weight_only_ptq"}:
            algorithm = "minmax"
        else:
            algorithm = None

    return {
        "quant_cfg": quant_cfg,
        "algorithm": algorithm,
    }
