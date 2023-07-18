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

"""Export Optical Inspection model to ONNX."""
import os
import torch

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.optical_inspection.config.default_config import OIExperimentConfig
from nvidia_tao_pytorch.cv.optical_inspection.model.pl_oi_model import OpticalInspectionModel
from nvidia_tao_pytorch.cv.optical_inspection.utils.common_utils import check_and_create

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment", schema=OIExperimentConfig
)
def main(cfg: OIExperimentConfig) -> None:
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
        run_export(cfg)
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


def run_export(args):
    """Wrapper to run export of tlt models.

    Args:
        args (dict): Dictionary of parsed arguments to run export.

    Returns:
        No explicit returns.
    """
    if args.export.results_dir is not None:
        results_dir = args.export.results_dir
    else:
        results_dir = os.path.join(args.results_dir, "export")
    check_and_create(results_dir)

    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            append=True
        )
    )
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting Optical Inspection export"
    )

    gpu_id = args.export.gpu_id
    torch.cuda.set_device(gpu_id)
    model_path = args.export.checkpoint
    on_cpu = args.export.on_cpu
    key = args.encryption_key
    output_file = args.export.onnx_file
    batch_size = args.export.batch_size
    input_channel = args.export.input_channel
    input_width = args.export.input_width
    input_height = args.export.input_height
    opset_version = args.export.opset_version
    do_constant_folding = args.export.do_constant_folding
    experiment_config = args

    if batch_size is None or batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    # Set default output filename if the filename
    # isn't provided over the command line.
    if output_file is None:
        split_name = os.path.splitext(model_path)[0]
        output_file = "{}.onnx".format(split_name)
    assert not os.path.exists(output_file), "Default output file {} already "\
        "exists.".format(output_file)

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # load model
    pl_model = OpticalInspectionModel.load_from_checkpoint(
        model_path,
        map_location="cpu",
        experiment_spec=experiment_config,
        export=True
    )
    model = pl_model.model
    model.eval()
    model.cuda()

    output_names = ["output_1", "output_2"]
    input_names = ["input_1", "input_2"]
    if on_cpu:
        dummy_input_1 = torch.randn(
            input_batch_size, input_channel, input_height, input_width, device="cpu")
        dummy_input_2 = torch.randn(
            input_batch_size, input_channel, input_height, input_width, device="cpu")
    else:
        dummy_input_1 = torch.randn(
            input_batch_size, input_channel, input_height, input_width, device="cuda")
        dummy_input_2 = torch.randn(
            input_batch_size, input_channel, input_height, input_width, device="cuda")
    dummy_input = (dummy_input_1, dummy_input_2)

    if batch_size is None or batch_size == -1:
        dynamic_axes = {
            "input_1": {0: "batch"},
            "input_2": {0: "batch"}
        }
    else:
        dynamic_axes = None

    # export
    torch.onnx.export(model,
                      dummy_input,
                      output_file,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=opset_version,
                      do_constant_folding=do_constant_folding,
                      verbose=True)


if __name__ == "__main__":
    main()
