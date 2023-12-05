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

"""Export re-identification model to ONNX."""

import os
import onnx
import torch
import onnx_graphsurgeon as gs

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.re_identification.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.re_identification.model.pl_reid_model import ReIdentificationModel
from nvidia_tao_pytorch.cv.pose_classification.utils.common_utils import check_and_create
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.utilities import update_results_dir


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """CLI wrapper to run export.

    This function serves as the entry point for the export script.
    It loads the experiment specification, updates the results directory,
    and calls the 'run_export' function. It also handles various exceptions
    and logs the export status.

    Args:
        cfg (ExperimentConfig): The experiment configuration retrieved from the Hydra configuration files.

    Raises:
        KeyboardInterrupt: If the process was interrupted manually.
        SystemExit: If the system or the program initiated the exit.
        Exception: For any other type of exception that occurred.
    """
    try:
        cfg = update_results_dir(cfg, task="export")
        run_export(cfg, results_dir=cfg.results_dir)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Export finished successfully"
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


def run_export(args, results_dir):
    """Run the export of the pose classification model to ONNX.

    This function handles the export process, including loading the model,
    creating dummy input, and exporting the model to an ONNX file.
    It also performs encryption on the ONNX file.

    Args:
        args (dict): Dictionary of parsed arguments to run export.
        results_dir (str): Directory to output results.

    Raises:
        AssertionError: If the default output file already exists.
        Exception: If any error occurs during the export process.
    """
    check_and_create(results_dir)

    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting Re-identification export")

    gpu_id = args['export']['gpu_id']
    torch.cuda.set_device(gpu_id)
    model_path = args['export']['checkpoint']

    # Parsing command line arguments.
    key = args['encryption_key']
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    onnx_file = args['export']['onnx_file']
    experiment_config = args

    # Set default output filename if the filename
    # isn't provided over the command line.
    if onnx_file is None:
        split_name = os.path.splitext(model_path)[0]
        onnx_file = "{}.onnx".format(split_name)

    assert not os.path.exists(onnx_file), "Default output file {} already "\
        "exists.".format(onnx_file)

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(onnx_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # load model
    pl_model = ReIdentificationModel.load_from_checkpoint(experiment_config["export"]["checkpoint"],
                                                          map_location="cpu",
                                                          experiment_spec=experiment_config,
                                                          prepare_for_training=False,
                                                          export=True)
    model = pl_model.model
    if "swin" in experiment_config.model.backbone:
        model.load_param(experiment_config["export"]["checkpoint"])

    model.eval()
    model.cuda()

    input_names = ["input"]
    output_names = ["fc_pred"]

    # create dummy input
    dummy_input = torch.randn(1, experiment_config["model"]["input_channels"],
                              experiment_config["model"]["input_height"], experiment_config["model"]["input_width"]).cuda()
    dynamic_axes = {"input": {0: "batch"}, "fc_pred": {0: "batch", 1: "embedding_size"}}
    # export
    torch.onnx.export(model,
                      dummy_input,
                      onnx_file,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      verbose=True)

    #  Remove excess layers from the backbone's last layer
    if "swin" in experiment_config.model.backbone:
        graph = gs.import_onnx(onnx.load(onnx_file))
        graph.outputs = graph.outputs[:-4]
        onnx.save(gs.export_onnx(graph), onnx_file)


if __name__ == "__main__":
    main()
