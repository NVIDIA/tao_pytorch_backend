# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Export Grounding DINO model to ONNX."""

import os
import torch

from nvidia_tao_core.config.grounding_dino.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utilities import get_nvdsinfer_yaml
from nvidia_tao_pytorch.cv.grounding_dino.utils.onnx_export import ONNXExporter
from nvidia_tao_pytorch.cv.grounding_dino.model.pl_gdino_model import GDINOPlModel
from nvidia_tao_pytorch.cv.grounding_dino.types.grounding_dino_nvdsinfer import GroundingDINONvDSInferConfig
from nvidia_tao_pytorch.cv.grounding_dino.types.grounding_dino_preprocess import GroundingDINOPreprocessConfig
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def serialize_nvdsinfer_config(output_root, output_file, input_shape, output_names, input_width, input_height, input_channel):
    """Serialize nvdsinfer configuration files.

    Args:
        output_root (str): Directory to save the configuration files
        output_file (str): Path to the ONNX model file
        input_shape (tuple): Input shape of the model
        output_names (list): List of output tensor names
        input_width (int): Input width
        input_height (int): Input height
        input_channel (int): Input channel

    Returns:
        tuple: Paths to the generated nvdsinfer and preprocess config files
    """
    nvdsinfer_yaml_file = os.path.join(output_root, "nvdsinfer_config.yaml")
    logging.info("Serializing the deepstream config to {}".format(nvdsinfer_yaml_file))

    num_classes = None
    labels_file = None
    nvdsinfer_config = get_nvdsinfer_yaml(
        GroundingDINONvDSInferConfig,
        num_classes,
        labels_file,
        output_file,
        input_shape,
        output_names
    )
    with open(nvdsinfer_yaml_file, "w") as f:
        f.write(nvdsinfer_config)

    nvdsinfer_preprocess_file = os.path.join(output_root, "nvdspreprocess_config.yaml")
    preprocess_config = GroundingDINOPreprocessConfig()
    preprocess_config.property.tensor_name = "inputs"
    preprocess_config.property.processing_width = input_width
    preprocess_config.property.processing_height = input_height
    preprocess_config.property.network_input_shape = [1, input_channel, input_height, input_width]

    with open(nvdsinfer_preprocess_file, "w") as f:
        f.write(str(preprocess_config))

    return nvdsinfer_yaml_file, nvdsinfer_preprocess_file


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="export", schema=ExperimentConfig
)
@monitor_status(name="Grounding DINO", mode="export")
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

    output_file = experiment_config.export.onnx_file
    input_channel = experiment_config.export.input_channel
    input_width = experiment_config.export.input_width
    input_height = experiment_config.export.input_height
    input_shape = [input_channel, input_height, input_width]
    serialize_nvdsinfer = experiment_config.export.serialize_nvdsinfer
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

    input_names = ["inputs", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"]
    output_names = ["pred_logits", "pred_boxes"]

    caption = "the running dog ."
    if serialize_nvdsinfer:
        serialize_nvdsinfer_config(
            output_root,
            output_file,
            input_shape,
            output_names,
            input_width,
            input_height,
            input_channel
        )

    # load model
    pl_model = GDINOPlModel.load_from_checkpoint(model_path,
                                                 map_location='cpu' if on_cpu else 'cuda',
                                                 experiment_spec=experiment_config,
                                                 export=True,
                                                 cap_lists=["the running dog"])
    model = pl_model.model
    model.eval()
    if not on_cpu:
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)

    # create dummy input
    tokenized, _, position_ids, text_self_attention_masks = pl_model.tokenize_captions([caption] * input_batch_size, pad_to_max=True)
    input_ids = tokenized["input_ids"].to(device)
    position_ids = position_ids.to(device)
    token_type_ids = tokenized["token_type_ids"].to(device)
    attention_mask = tokenized["attention_mask"].bool().to(device)
    text_token_mask = text_self_attention_masks.to(device)

    dummy_input = torch.randn(input_batch_size, input_channel, input_height, input_width).to(device)

    args = (dummy_input, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask)

    onnx_export = ONNXExporter()
    onnx_export.export_model(model,
                             batch_size,
                             output_file,
                             args,
                             input_names=input_names,
                             opset_version=opset_version,
                             output_names=output_names,
                             do_constant_folding=False,
                             verbose=experiment_config.export.verbose)
    onnx_export.check_onnx(output_file)
    onnx_export.onnx_change(output_file)

    logging.info(f"ONNX file stored at {output_file}")


if __name__ == "__main__":
    main()
