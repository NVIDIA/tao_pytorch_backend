# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
"""Classification model utilities.

Provide helpers to construct a :class:`ClassifierPlModel` from experiment
configuration for different tasks with support for loading quantized checkpoints.
"""

import torch
from nvidia_tao_pytorch.core.tlt_logging import logging

from nvidia_tao_pytorch.cv.classification_pyt.model.classifier_pl_model import ClassifierPlModel
from nvidia_tao_pytorch.core.quantization.quantizer import ModelQuantizer


def create_model_from_config(experiment_config, model_path: str, task: str = "evaluate"):
    """Construct a Classification Lightning model from configuration.

    Supports both standard checkpoints and quantized checkpoints. When quantized,
    the function prepares a quantized model using the selected backend and loads a
    compatible state dict.

    Parameters
    ----------
    experiment_config
        Experiment configuration instance for classification from ``tao-core``.
    model_path : str
        Filesystem path to the model artifact (e.g., ``.pth``). For ModelOpt
        artifacts, the state dict is expected under the key ``"model_state_dict"``.
    task : {"evaluate", "inference", "export"}, optional
        Task context to choose the right flags from the configuration, by default
        ``"evaluate"``.

    Returns
    -------
    ClassifierPlModel
        LightningModule that wraps the backbone :class:`torch.nn.Module`.

    Raises
    ------
    ValueError
        If an unsupported task is provided or if a non-quantized checkpoint fails to load.
    """
    logging.info(f"Creating model from config for task: {task}, model_path: {model_path}")

    if task == "evaluate":
        is_quantized = experiment_config.evaluate.is_quantized
    elif task == "inference":
        is_quantized = experiment_config.inference.is_quantized
    elif task == "export":
        is_quantized = experiment_config.export.is_quantized
    else:
        logging.error(f"Unsupported task: {task}")
        raise ValueError(f"Unsupported task: {task}")

    if is_quantized:
        logging.info("is_quantized is True. Building quantized model.")
        model = ClassifierPlModel(experiment_config)
        quantizer = ModelQuantizer(experiment_config.quantize)

        logging.info("Applying quantizer to model to load quantized model state dict.")
        model = quantizer.quantize_model(model)

        logging.debug(f"Loading quantized model state dict from {model_path}")

        quantized_model_state_dict = torch.load(model_path, map_location="cpu")

        if experiment_config.quantize.backend == "modelopt":
            quantized_model_state_dict = quantized_model_state_dict["model_state_dict"]

        quantized_model_state_dict = {f"model.{k}": v for k, v in quantized_model_state_dict.items()}
        logging.debug("Loading state dict into quantized model.")
        model.load_state_dict(quantized_model_state_dict)
        logging.info("Quantized model loaded successfully.")

    else:
        # build model and load from the given checkpoint
        try:
            logging.info(f"Loading model from checkpoint: {model_path}")
            model = ClassifierPlModel.load_from_checkpoint(
                model_path,
                map_location="cpu",
                experiment_spec=experiment_config
            )
            logging.info("Model loaded from checkpoint successfully.")
        except Exception as e:
            logging.error(f"Failed to load model from checkpoint: {e}")
            raise ValueError(
                f"Failed to load model from checkpoint: {e}. If the model was quantized, please set is_quantized to true."
            )

    return model
