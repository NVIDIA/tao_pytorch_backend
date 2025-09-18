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

"""Script to export a DepthNet model to ONNX format."""

import os
import torch

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import get_pl_module
from nvidia_tao_core.config.depth_net.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.mae.scripts.export import create_onnx_model

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="DepthNet", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """Entry point for DepthNet model export process.

    This function serves as the main entry point for exporting a trained MAE (Masked Autoencoder)
    model. It configures the PyTorch backend settings and delegates the actual export process
    to the run_export function.

    Args:
        cfg (ExperimentConfig): Hydra configuration object containing all export parameters.
            This object is automatically populated by Hydra based on the experiment spec
            and command line arguments.

    Note:
        This function is decorated with @hydra_runner and @monitor_status to handle
        configuration loading and workflow monitoring respectively. The actual export
        logic is implemented in the run_export function.

    Example:
        >>> # The function is typically called via command line:
        >>> # python export.py export.onnx_file=model.onnx export.input_channel=3
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_export(cfg)


def run_export(experiment_config: ExperimentConfig) -> None:
    """Execute the DepthNet model export process to ONNX format.

    This function handles the core export process for a trained MAE (Masked Autoencoder) model.
    It processes the experiment configuration, sets up the model with proper encryption,
    and exports it to ONNX format with the specified parameters.

    Args:
        experiment_config (ExperimentConfig): Configuration object containing export parameters:
            - export.gpu_id: GPU device ID to use for export
            - export.checkpoint: Path to the model checkpoint file
            - encryption_key: Key for model encryption/decryption
            - export.onnx_file: Output path for the ONNX model
            - export.input_channel: Number of input channels (e.g., 3 for RGB)
            - export.input_width: Input image width
            - export.input_height: Input image height
            - export.opset_version: ONNX opset version for export
            - export.batch_size: Batch size for export (defaults to 1 if None or -1)
            - export.on_cpu: Whether to perform export on CPU instead of GPU

    Raises:
        AssertionError: If the output ONNX file already exists at the specified path.
        RuntimeError: If model loading fails or if there are issues during ONNX export.
        ValueError: If required configuration parameters are missing or invalid.

    Note:
        The function handles both CPU and GPU exports based on the configuration.
        For GPU exports, it sets the appropriate CUDA device before proceeding.

    Example:
        >>> config = ExperimentConfig()
        >>> config.export.onnx_file = "model.onnx"
        >>> config.export.input_channel = 3
        >>> config.export.input_width = 924
        >>> config.export.input_height = 518
        >>> run_export(config)
    """
    # Convert DictConfig to ExperimentConfig
    gpu_id = experiment_config.export.gpu_id
    torch.cuda.set_device(gpu_id)

    # Parsing command line arguments.
    model_path = experiment_config.export.checkpoint
    key = experiment_config.encryption_key

    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    output_file = experiment_config.export.onnx_file
    input_channel = experiment_config.export.input_channel
    input_width = experiment_config.export.input_width
    input_height = experiment_config.export.input_height
    input_shape = [input_channel, input_height, input_width]
    opset_version = experiment_config.export.opset_version
    batch_size = experiment_config.export.batch_size
    on_cpu = experiment_config.export.on_cpu
    if batch_size is None or batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    # Set default output filename if the filename.
    if output_file is None:
        split_name = os.path.splitext(model_path)[0]
        output_file = "{}.onnx".format(split_name)

    # Create output directory
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Setting input/output tensor names.
    input_names = ['images']
    output_names = ['outputs']

    # Load model
    pl_model = get_pl_module(experiment_config).load_from_checkpoint(
        model_path,
        experiment_spec=experiment_config,
        export=True,  # to use regular Attention instead of Memory-efficient Attention for export
        map_location='cpu'
    )

    model = pl_model.model
    try:
        dynamic_axis = False
        if batch_size == -1:
            dynamic_axis = True
        create_onnx_model(
            model,
            input_shape,
            input_batch_size,
            output_file,
            input_names,
            output_names,
            on_cpu=on_cpu,
            dynamic_axis=dynamic_axis,
            opset_version=opset_version,
        )
        logging.info("ONNX model saved to {output_file}".format(output_file=output_file))
    except Exception as e:
        logging.error(f"Error exporting model: {e}")
        raise e


if __name__ == "__main__":
    main()
