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

"""Export action recognition model to ONNX."""

import os
import torch
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.action_recognition.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.action_recognition.model.pl_ar_model import ActionRecognitionModel
from nvidia_tao_pytorch.cv.action_recognition.utils.common_utils import check_and_create
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """CLI wrapper to run export.
    This function parses the command line interface for tlt-export, instantiates the respective
    exporter and serializes the trained model to an etlt file. The tools also runs optimization
    to the int8 backend.

    Args:
        cl_args(list): Arguments to parse.

    Returns:
        No explicit returns.
    """
    try:
        if cfg.export.results_dir is not None:
            results_dir = cfg.export.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "export")
        run_export(cfg, output_dir=results_dir)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Export finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Export was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e


def run_export(args, output_dir):
    """Wrapper to run export of tlt models.

    Args:
        args (dict): Dictionary of parsed arguments to run export.

    Returns:
        No explicit returns.
    """
    check_and_create(output_dir)

    # Set status logging
    status_file = os.path.join(output_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting Action recognition export")

    gpu_id = args.export.gpu_id
    torch.cuda.set_device(gpu_id)
    # Parsing command line arguments.
    model_path = args["export"]['checkpoint']
    key = args['encryption_key']
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    # data_type = args['data_type']
    output_file = args["export"]['onnx_file']
    experiment_config = args

    # Set default output filename if the filename
    # isn't provided over the command line.
    if output_file is None:
        split_name = os.path.splitext(model_path)[0]
        output_file = "{}.onnx".format(split_name)

    # Warn the user if an exported file already exists.
    assert not os.path.exists(output_file), "Default output file {} already "\
        "exists".format(output_file)

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # load model
    pl_model = ActionRecognitionModel.load_from_checkpoint(model_path,
                                                           map_location="cpu",
                                                           experiment_spec=experiment_config,
                                                           export=True)
    model = pl_model.model
    model.eval()
    model.cuda()

    model_type = experiment_config['model']['model_type']
    if model_type == "of":
        input_names = ["input_of"]
    elif model_type == "rgb":
        input_names = ["input_rgb"]
    elif model_type == "joint":
        input_names = ["input_rgb", "input_of"]
    else:
        raise ValueError("Wrong model type in the config")

    output_names = ["fc_pred"]

    # create dummy input
    output_shape = [experiment_config["model"]["input_height"],
                    experiment_config["model"]["input_width"]]
    rgb_seq_length = experiment_config['model']['rgb_seq_length']
    of_seq_length = experiment_config['model']['of_seq_length']
    input_type = experiment_config['model']['input_type']
    if input_type == "2d":
        if model_type == "of":
            dummy_input = torch.randn(1, 2 * of_seq_length,
                                      output_shape[0], output_shape[1]).cuda()
            dynamic_axes = {"input_of": {0: "batch"}, "fc_pred": {0: "batch"}}
        elif model_type == "rgb":
            dummy_input = torch.randn(1, 3 * rgb_seq_length,
                                      output_shape[0], output_shape[1]).cuda()
            dynamic_axes = {"input_rgb": {0: "batch"}, "fc_pred": {0: "batch"}}
        elif model_type == "joint":
            dummy_input = (torch.randn(1, 3 * rgb_seq_length,
                                       output_shape[0], output_shape[1]).cuda(),
                           torch.randn(1, 2 * of_seq_length,
                                       output_shape[0], output_shape[1]).cuda())
            dynamic_axes = {"input_rgb": {0: "batch"}, "input_of": {0: "batch"},
                            "fc_pred": {0: "batch"}}
        else:
            raise ValueError("Wrong model type in the config")
    elif input_type == "3d":
        if model_type == "of":
            dummy_input = torch.randn(1, 2, of_seq_length,
                                      output_shape[0], output_shape[1]).cuda()
            dynamic_axes = {"input_of": {0: "batch"}, "fc_pred": {0: "batch"}}
        elif model_type == "rgb":
            dummy_input = torch.randn(1, 3, rgb_seq_length,
                                      output_shape[0], output_shape[1]).cuda()
            dynamic_axes = {"input_rgb": {0: "batch"}, "fc_pred": {0: "batch"}}
        elif model_type == "joint":
            dummy_input = (torch.randn(1, 3, rgb_seq_length,
                                       output_shape[0], output_shape[1]).cuda(),
                           torch.randn(1, 2, of_seq_length,
                                       output_shape[0], output_shape[1]).cuda())
            dynamic_axes = {"input_rgb": {0: "batch"}, "input_of": {0: "batch"},
                            "fc_pred": {0: "batch"}}
        else:
            raise ValueError("Wrong model type in the config")

    # export
    torch.onnx.export(model,
                      dummy_input,
                      output_file,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      verbose=True)


if __name__ == "__main__":
    main()
