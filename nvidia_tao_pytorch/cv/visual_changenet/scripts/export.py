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

"""Export Visual ChangeNet model to ONNX."""

import os
import torch

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.utilities import encrypt_onnx
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule
from nvidia_tao_pytorch.cv.visual_changenet.utils.onnx_export import ONNXExporter
from nvidia_tao_pytorch.cv.visual_changenet.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlSegment
from nvidia_tao_pytorch.cv.visual_changenet.classification.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlClassifier

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="Visual ChangeNet", mode="export")
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
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_export(cfg)


def run_export(experiment_config):
    """Wrapper to run export of tlt models.

    Args:
        args (dict): Dictionary of parsed arguments to run export.

    Returns:
        No explicit returns.
    """
    gpu_id = experiment_config.export.gpu_id
    torch.cuda.set_device(gpu_id)

    # Parsing command line arguments.
    model_path = experiment_config.export.checkpoint
    key = experiment_config.encryption_key

    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    task = experiment_config.task
    output_file = experiment_config.export.onnx_file
    input_channel = experiment_config.export.input_channel
    input_width = experiment_config.export.input_width
    input_height = experiment_config.export.input_height
    opset_version = experiment_config.export.opset_version
    batch_size = experiment_config.export.batch_size
    on_cpu = experiment_config.export.on_cpu
    if batch_size is None or batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    # Set default output filename if the filename
    # isn't provided over the command line.
    if output_file is None:
        split_name = os.path.splitext(model_path)[0]
        output_file = "{}.onnx".format(split_name)

    # Warn the user if an exported file already exists.
    assert not os.path.exists(output_file), "Default onnx file {} already "\
        "exists".format(output_file)

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    assert task in ['segment', 'classify'], "Visual ChangeNet only supports 'segment' and 'classify' tasks."
    if task == 'classify':
        dm = OIDataModule(experiment_config)
        # load model
        cf_model = ChangeNetPlClassifier.load_from_checkpoint(model_path,
                                                              map_location="cpu" if on_cpu else 'cuda',
                                                              experiment_spec=experiment_config,
                                                              dm=dm,
                                                              export=True)

        output_names = ["output"]
        input_names = ["input_1", "input_2"]

    elif task == 'segment':
        # load model
        cf_model = ChangeNetPlSegment.load_from_checkpoint(model_path,
                                                           map_location="cpu" if on_cpu else 'cuda',
                                                           experiment_spec=experiment_config,
                                                           export=True)

        input_names = ['input0', 'input1']
        output_names = ["output0", "output1", 'output2', 'output3', 'output_final']
    else:
        raise NotImplementedError('Only tasks supported by Visual ChangeNet are: "segment" and "classify"')

    model = cf_model.model
    model.eval()
    if not on_cpu:
        model.cuda()

    # create dummy input
    if on_cpu:
        dummy_input0 = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cpu')
        dummy_input1 = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cpu')
    else:
        dummy_input0 = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda')
        dummy_input1 = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda')
    dummy_input = (dummy_input0, dummy_input1)

    if output_file.endswith('.etlt'):
        tmp_onnx_file = output_file.replace('.etlt', '.onnx')
    else:
        tmp_onnx_file = output_file

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, batch_size,
                             tmp_onnx_file,
                             dummy_input,
                             input_names=input_names,
                             opset_version=opset_version,
                             output_names=output_names,
                             do_constant_folding=True,
                             verbose=experiment_config.export.verbose,
                             task=task)

    onnx_export.check_onnx(output_file)

    if output_file.endswith('.etlt') and key:
        # encrypt the onnx if and only if key is provided and output file name ends with .etlt
        encrypt_onnx(tmp_file_name=tmp_onnx_file,
                     output_file_name=output_file,
                     key=key)

        os.remove(tmp_onnx_file)
    print(f"ONNX file stored at {output_file}")


if __name__ == "__main__":
    main()
