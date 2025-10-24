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
import onnx

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import get_pl_module
from nvidia_tao_core.config.depth_net.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.mae.scripts.export import create_onnx_model

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTOCAST = torch.amp.autocast


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


def mono_onnx_export(model,
                     input_shape,
                     input_batch_size,
                     output_file,
                     on_cpu,
                     opset_version,
                     valid_iters=22,
                     dynamic_axis=False):
    """
    Exports a monocular depth estimation model to the ONNX format.

    This function serves as a wrapper for the `create_onnx_model` utility, handling
    the specifics of exporting a monocular depth estimation model. It defines the
    input and output tensor names and determines whether to enable dynamic axes
    based on the provided `batch_size`. If `batch_size` is -1, the exported ONNX
    model will have a dynamic batch dimension, allowing for flexible input
    sizes at inference time. The export process is delegated to
    `create_onnx_model`, which handles the core ONNX conversion logic.

    Args:
        model (torch.nn.Module): The PyTorch monocular depth estimation model to
                                  be exported.
        input_shape (list or tuple): The shape of a single input image tensor,
                                     excluding the batch dimension. Example:
                                     `[3, 256, 256]` for a 256x256 color image.
        input_batch_size (int): The batch size to be used for the dummy input
                                during the export process. This is the concrete
                                batch size used to create the dummy tensor,
                                even if `batch_size` is set to -1 for a
                                dynamic axis.
        output_file (str): The path to the output ONNX file (e.g., 'model.onnx').
        on_cpu (bool): If True, the dummy input tensors will be created on the CPU.
                       If False, they will be created on the GPU (CUDA).
        opset_version (int): The ONNX operator set version to use for the export.

    Raises:
        Exception: If the underlying `create_onnx_model` function or any part
                   of the export process fails, an exception is raised with
                   a detailed error message.
    """
    input_names = ['images']
    output_names = ['outputs']
    try:
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


def stereo_onnx_export(model,
                       input_shape,
                       input_batch_size,
                       output_file,
                       on_cpu,
                       opset_version,
                       valid_iters=22,
                       dynamic_axis=True):
    """
    Exports a stereo depth estimation model to the ONNX format.

    This function prepares a dummy input tensor and then uses `torch.onnx.export` to
    convert the PyTorch model into an ONNX representation. The exported model
    includes dynamic axes for batch size, height, and width, making it
    flexible for different input sizes at inference time. After export, it
    verifies the integrity of the generated ONNX file using `onnx.checker.check_model`.

    Args:
        model (torch.nn.Module): The PyTorch stereo depth estimation model to be exported.
                                  The model should accept two images (left and right)
                                  and a set of additional arguments as input.
        input_shape (list or tuple): The shape of a single input image,
                                     excluding the batch dimension.
                                     Example: `[3, 256, 256]` for a color image
                                     of size 256x256.
        input_batch_size (int): The batch size to be used for the dummy input.
                                This value is used to set the first dimension of the
                                input tensors, but the exported ONNX model will have
                                a dynamic batch size.
        output_file (str): The path to the output ONNX file (e.g., 'model.onnx').
        on_cpu (bool): If True, the dummy input tensors will be created on the CPU.
                       If False, they will be created on the GPU (CUDA).
        opset_version (int): The ONNX operator set version to use for the export.
                             A higher version may support more recent operations.

    Raises:
        Exception: If the ONNX export or the subsequent model check fails,
                   an exception is raised, providing details about the error.

    Notes:
        - The function uses `AUTOCAST('cuda', enabled=True)` to ensure the dummy
          inputs are created with mixed-precision if available, which can
          be important for models trained with `torch.autocast`.
        - The `dynamic_axes` argument is crucial for creating a model that
          can handle variable-sized inputs, which is a common requirement for
          computer vision models. It maps the dimensions of the input/output
          tensors to descriptive names like 'batch_size', 'height', and 'width'.
        - The dummy input arguments (`args=(dummy_input1, dummy_input2, 4, ...)`
          are specific to the model's forward method. This structure should be
          modified if the model's signature changes.
    """
    input_names = ['left_image', 'right_image', 'iters',
                   'flow_init', 'test_mode', 'low_mem', 'init_disp']
    output_names = ["disparity"]

    input_shape.insert(0, input_batch_size)
    with AUTOCAST('cuda', enabled=True):
        if on_cpu:
            dummy_input1 = torch.rand(input_shape, device='cpu')
            dummy_input2 = torch.rand(input_shape, device='cpu')
        else:
            dummy_input1 = torch.rand(input_shape, device='cuda')
            dummy_input2 = torch.rand(input_shape, device='cuda')
    try:
        # Only use dynamic_axes if dynamic_axis is True
        axes_config = None
        if dynamic_axis:
            axes_config = {'left_image': {0: 'batch_size', 2: 'height', 3: 'width'},
                           'right_image': {0: 'batch_size', 2: 'height', 3: 'width'},
                           'disparity': {0: 'batch_size'}}

        torch.onnx.export(model,
                          args=(dummy_input1, dummy_input2, valid_iters, None, True, False, None),
                          f=output_file,
                          input_names=input_names,
                          opset_version=opset_version,
                          output_names=output_names,
                          do_constant_folding=True,
                          verbose=True,
                          dynamic_axes=axes_config)

        # Verify ONNX exported correctly.
        loaded_model = onnx.load(output_file)
        onnx.checker.check_model(loaded_model)

    except Exception as e:
        logging.error(f"Error exporting model: {e}")
        raise e


def onnx_model_export(model_type):
    """ Factory function to export ONNX for mono and stereo models.
    Args:
        model_type (str): the model type to be exported.

    Returns:
        an export function for either modes.

    """
    onnx_export_method = {
        'metricdepthanything': mono_onnx_export,
        'relativedepthanything': mono_onnx_export,
        'foundationstereo': stereo_onnx_export}

    if model_type.lower() not in onnx_export_method:
        raise (NotImplementedError(f'{model_type} does not have onnx export implemented!'))
    return onnx_export_method[model_type.lower()]


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
            - export.valid_iters: Number of GPU valid iterations to refine disparity

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
    valid_iters = experiment_config.export.valid_iters

    if batch_size is None or batch_size == -1:
        input_batch_size = 1
        dynamic_axis = True
    else:
        input_batch_size = batch_size
        dynamic_axis = False

    logging.info(f"Input batch size: {input_batch_size}")
    logging.info(f"Dynamic axis: {dynamic_axis}")

    device = 'cpu'
    if not on_cpu:
        device = 'cuda'

    # Set default output filename if the filename.
    if output_file is None:
        split_name = os.path.splitext(model_path)[0]
        output_file = "{}.onnx".format(split_name)

    # Create output directory
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Load model
    pl_model = get_pl_module(experiment_config).load_from_checkpoint(
        model_path,
        experiment_spec=experiment_config,
        export=True,  # to use regular Attention instead of Memory-efficient Attention for export
        map_location=device
    )

    model = pl_model.model
    onnx_model_export(experiment_config.model.model_type)(model,
                                                          input_shape,
                                                          input_batch_size,
                                                          output_file,
                                                          on_cpu,
                                                          opset_version,
                                                          valid_iters=valid_iters,
                                                          dynamic_axis=dynamic_axis)


if __name__ == "__main__":
    main()
