#!/usr/bin/env python3
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

"""RT-DETR model utilities.

Provides helpers to build an :class:`RTDETRPlModel` from an experiment configuration
for different tasks and supports loading quantized checkpoints produced by ModelOpt or
other backends.
"""
from nvidia_tao_pytorch.core.tlt_logging import logging
import torch

from nvidia_tao_pytorch.cv.rtdetr.model.pl_rtdetr_model import RTDETRPlModel
from nvidia_tao_pytorch.core.quantization.quantizer import ModelQuantizer


def create_model_from_config(experiment_config, model_path: str, task: str = "evaluate"):
    """Construct a RT-DETR Lightning model from configuration.

    Supports both standard checkpoints and quantized checkpoints. When quantized,
    the function prepares a quantized model using the selected backend and loads a
    compatible state dict.

    Parameters
    ----------
    experiment_config
        Experiment configuration instance from ``tao-core`` for RT-DETR.
    model_path : str
        Filesystem path to the model artifact (e.g., ``.pth``). For ModelOpt
        artifacts, the state dict is expected under the key ``"model_state_dict"``.
    task : {"evaluate", "inference", "export"}, optional
        Task context to choose the right flags from the configuration, by default
        ``"evaluate"``.

    Returns
    -------
    RTDETRPlModel
        LightningModule that wraps the RT-DETR :class:`torch.nn.Module`. For export,
        the module is constructed with ``export=True``.

    Raises
    ------
    ValueError
        If an unsupported task is provided or if a non-quantized checkpoint fails to load.
    """
    logging.info(f"Creating RT-DETR model from config for task: {task}, model_path: {model_path}")

    if task == "evaluate":
        is_quantized = experiment_config.evaluate.is_quantized
        export_flag = False
    elif task == "inference":
        is_quantized = experiment_config.inference.is_quantized
        export_flag = False
    elif task == "export":
        is_quantized = experiment_config.export.is_quantized
        export_flag = True
    else:
        logging.error(f"Unsupported task: {task}")
        raise ValueError(f"Unsupported task: {task}")

    if is_quantized:
        logging.info("is_quantized is True. Building quantized model.")
        # Build the LightningModule and quantize it so that quantized state dict can load.
        model = RTDETRPlModel(experiment_config, export=export_flag)
        quantizer = ModelQuantizer(experiment_config.quantize)

        logging.info("Applying quantizer to model to load quantized model state dict.")
        model = quantizer.quantize_model(model)

        logging.debug(f"Loading quantized model state dict from {model_path}")

        quantized_model_state_dict = torch.load(model_path, map_location="cpu")

        # ModelOpt backend saves a structured artifact; retrieve the bare model state dict.
        try:
            backend = experiment_config.quantize.backend
        except Exception:
            backend = None

        if backend == "modelopt" and isinstance(quantized_model_state_dict, dict) and "model_state_dict" in quantized_model_state_dict:
            quantized_model_state_dict = quantized_model_state_dict["model_state_dict"]

        # Prefix keys with "model." to match the Lightning module structure.
        quantized_model_state_dict = {f"model.{k}": v for k, v in quantized_model_state_dict.items()}
        logging.debug("Loading state dict into quantized model.")
        model.load_state_dict(quantized_model_state_dict)
        logging.info("Quantized RT-DETR model loaded successfully.")
    else:
        # Build model and load from the given checkpoint
        logging.info(f"Loading RT-DETR model from checkpoint: {model_path}")
        try:
            model = RTDETRPlModel.load_from_checkpoint(
                model_path,
                map_location="cpu",
                experiment_spec=experiment_config,
                export=export_flag,
            )
            logging.info("RT-DETR model loaded from checkpoint successfully.")
        except Exception as e:
            logging.error(f"Failed to load model from checkpoint: {e}")
            raise ValueError(
                f"Failed to load model from checkpoint: {e}. If the model was quantized, please set is_quantized to true."
            )

    return model
