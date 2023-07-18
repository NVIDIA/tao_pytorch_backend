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

"""Export metric-learning recognition model to ONNX."""

import os
import tempfile
import torch
from onnxsim import simplify
import onnx

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.metric_learning_recognition.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.metric_learning_recognition.model.pl_ml_recog_model import MLRecogModel
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.cv.metric_learning_recognition.utils.decorators import monitor_status
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_export(args):
    """Wrapper to run export of .pth checkpoints.

    Args:
        args (DictConfig): Configuration dictionary
    """
    experiment_config = args

    # no need to check `else` as it's verified in the decorator already
    if experiment_config['export']["results_dir"]:
        results_dir = experiment_config['export']["results_dir"]
    elif experiment_config["results_dir"]:
        results_dir = os.path.join(experiment_config["results_dir"], "export")

    if experiment_config['export']["on_cpu"]:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
        gpu_id = experiment_config.export.gpu_id
        torch.cuda.set_device(gpu_id)
    else:
        error_msg = "No GPU available for export."
        raise ValueError(error_msg)

    checkpoint = experiment_config["export"]["checkpoint"]
    if checkpoint is not None:
        status_logging.get_status_logger().write(
            message=f"Loading checkpoint: {experiment_config['export']['checkpoint']}",
            status_level=status_logging.Status.STARTED)
        pl_model = MLRecogModel.load_from_checkpoint(experiment_config["export"]["checkpoint"],
                                                     map_location="cpu",
                                                     experiment_spec=experiment_config,
                                                     results_dir=results_dir,
                                                     subtask="export")
        # Set default output filename if the filename
        # isn't provided over the command line.
        if experiment_config['export']['onnx_file'] is None:
            split_name = os.path.splitext(os.path.basename(checkpoint))[0]
            output_file = os.path.join(results_dir, f"{split_name}.onnx")
        else:
            output_file = experiment_config['export']['onnx_file']

    else:
        pl_model = MLRecogModel(experiment_config,
                                results_dir,
                                subtask="export")
        if experiment_config['export']['onnx_file'] is None:
            output_file = os.path.join(results_dir, "metric_learning_recognition.onnx")
        else:
            output_file = experiment_config['export']['onnx_file']

    assert not os.path.exists(output_file), f"output file {output_file} "\
        "already exists."

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    model = pl_model.model
    model.eval()
    model.to(device)

    input_names = ["input"]
    output_names = ["fc_pred"]

    # create dummy input
    input_channels = experiment_config["model"]["input_channels"]
    input_width = experiment_config["model"]["input_width"]
    input_height = experiment_config["model"]["input_height"]
    batch_size = experiment_config["export"]["batch_size"]
    if batch_size == -1:
        dynamic_axes = {"input": {0: "batch"}, "fc_pred": {0: "batch"}}
        dummy_input = torch.randn(
            1,  input_channels, input_width, input_height).to(device)
    elif batch_size >= 1:
        dynamic_axes = None
        dummy_input = torch.randn(
            batch_size,  input_channels, input_width, input_height).to(device)
    else:
        raise ValueError("`export.batch_size` must be greater than 0 or -1.")

    # export
    status_logging.get_status_logger().write(
        message="Exporting model to ONNX",
        status_level=status_logging.Status.STARTED)
    os_handle, tmp_onnx_file = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)
    torch.onnx.export(model,
                      dummy_input,
                      tmp_onnx_file,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=experiment_config["export"]["opset_version"],
                      dynamic_axes=dynamic_axes,
                      verbose=experiment_config["export"]["verbose"])

    # add simplification
    status_logging.get_status_logger().write(
        message="Simplifying ONNX model",
        status_level=status_logging.Status.STARTED)
    simplified_model, _ = simplify(
        tmp_onnx_file,
        test_input_shapes={'input': (1, input_channels, input_width, input_height)},
        check_n=3)
    onnx.save(simplified_model, output_file)
    status_logging.get_status_logger().write(
        message=f"ONNX model saved at {output_file}",
        status_level=status_logging.Status.SUCCESS)


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="export", schema=ExperimentConfig
)
@monitor_status(mode="export")
def main(cfg: ExperimentConfig) -> None:
    """CLI wrapper to run export.

    This function parses the command line interface for tlt-export, instantiates the respective
    exporter and serializes the trained model to an onnx file. The tools also runs optimization
    to the int8 backend.

    Args:
        cl_args(list): Arguments to parse.

    Returns:
        No explicit returns.
    """
    obfuscate_logs(cfg)

    run_export(cfg)


if __name__ == "__main__":
    main()
