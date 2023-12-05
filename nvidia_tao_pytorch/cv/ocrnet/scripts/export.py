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

"""
Export OCRNet script.
"""
import os
import argparse
import tempfile
import onnx_graphsurgeon as gs
import onnx

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.ocrnet.config.default_config import ExperimentConfig


@gs.Graph.register()
def replace_with_avgpool2d(self, inputs, outputs, kernel_shape,
                           pads=[0, 0, 0, 0], strides=[1, 1]):
    """helper function to replace adaptive pool to avgpool2d.

    Args:
        inputs (torch.Tensor): The input onnx node.
        outputs (torch.Tensor): The output onnx node.
        kernel_shape (tuple): A tuple containing the height and width of the kernel.
        pads (list, optional): A list containing the padding values for the top, bottom, left, and right sides of the input. Defaults to [0, 0, 0, 0].
        strides (list, optional): A list containing the stride values for the height and width of the kernel. Defaults to [1, 1].
    """
    # Disconnect output nodes of all input tensors
    for inp in inputs:
        inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    attrs = {"ceil_mode": 0, "kernel_shape": kernel_shape, "pads": pads, "strides": strides}
    # Insert the new node.
    return self.layer(op="AveragePool", attrs=attrs,
                      inputs=inputs, outputs=outputs)


def export(opt):
    """Export the model according to option."""
    # @TODO(tylerz): Lazy import for correctly setting CUDA_VISIBLE_DEVICES
    import torch
    import torch.utils.data

    from nvidia_tao_pytorch.cv.ocrnet.utils.utils import (CTCLabelConverter,
                                                          AttnLabelConverter,
                                                          load_checkpoint)
    from nvidia_tao_pytorch.cv.ocrnet.model.model import Model, ExportModel
    from pytorch_quantization import nn as quant_nn
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    ckpt = load_checkpoint(opt.saved_model, key=opt.encryption_key, to_cpu=True)
    if not isinstance(ckpt, Model):
        model = Model(opt)
        state_dict = ckpt
        model.load_state_dict(state_dict)
        model = ExportModel(ocr_model=model, prediction_type=opt.Prediction)
    else:
        for name, m in ckpt.named_modules():
            if "quantizer" in name:
                m.use_fb_fake_quant = True
        model = ExportModel(ocr_model=ckpt, prediction_type=opt.Prediction)

    model = model.to(device)

    input_names = ["input"]
    output_names = ["output_id", "output_prob"]
    dummy_input = (torch.randn(13, opt.input_channel, opt.imgH, opt.imgW).to(device),
                   torch.LongTensor(13, opt.batch_max_length + 1).fill_(0).to(device))
    dynamic_axes = {"input": {0: "batch"}, "output_id": {0: "batch"}, "output_prob": {0: "batch"}}
    os_handle, tmp_file_name = tempfile.mkstemp()
    os.close(os_handle)
    output_file = tmp_file_name
    torch.onnx.export(model,
                      dummy_input,
                      output_file,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                      verbose=False,
                      opset_version=17,
                      do_constant_folding=True
                      )

    graph = gs.import_onnx(onnx.load(output_file))

    # Do a dummy inference to get visual feature height
    visual_feature = model.ocr_model.FeatureExtraction(dummy_input[0])
    feature_height = visual_feature.shape[2]

    for node in graph.nodes:
        if node.op == "adaptive_avg_pool2d":
            inp_tensor = [node.inputs[0]]
            # node.i(1, 0).outputs.clear()
            del node.inputs[1]
            oup_tensor = [node.outputs[0]]
            graph.replace_with_avgpool2d(inp_tensor, oup_tensor, kernel_shape=[1, feature_height])
            del node

    graph.cleanup()
    onnx.save(gs.export_onnx(graph), opt.output_file)

    os.remove(tmp_file_name)


def init_configs(experiment_spec: ExperimentConfig):
    """Pass the yaml config to argparse.Namespace"""
    parser = argparse.ArgumentParser()

    opt, _ = parser.parse_known_args()
    opt.encryption_key = experiment_spec.encryption_key
    opt.output_file = experiment_spec.export.onnx_file

    # 1. Init dataset params
    dataset_config = experiment_spec.dataset
    model_config = experiment_spec.model
    opt.batch_max_length = dataset_config.max_label_length
    opt.imgH = model_config.input_height
    opt.imgW = model_config.input_width
    opt.input_channel = model_config.input_channel
    if model_config.input_channel == 3:
        opt.rgb = True
    else:
        opt.rgb = False

    # load character list:
    # Don't convert the characters to lower case
    with open(dataset_config.character_list_file, "r") as f:
        characters = "".join([ch.strip() for ch in f.readlines()])
    opt.character = characters

    # 2. Init Model params
    opt.saved_model = experiment_spec.export.checkpoint
    if model_config.TPS:
        opt.Transformation = "TPS"
    else:
        opt.Transformation = "None"

    opt.FeatureExtraction = model_config.backbone
    opt.SequenceModeling = model_config.sequence
    opt.Prediction = model_config.prediction
    opt.num_fiducial = model_config.num_fiducial
    opt.output_channel = model_config.feature_channel
    opt.hidden_size = model_config.hidden_size

    opt.baiduCTC = False

    # 4. Init for Device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(experiment_spec.export.gpu_id)
    import torch
    opt.num_gpu = torch.cuda.device_count()

    return opt


def run_experiment(experiment_spec):
    """run experiment."""
    opt = init_configs(experiment_spec)
    # Set default output filename if the filename
    # isn't provided over the command line.
    if opt.output_file is None:
        split_name = os.path.splitext(opt.saved_model)[0]
        opt.output_file = "{}.etlt".format(split_name)

    # Warn the user if an exported file already exists.
    if os.path.exists(opt.output_file):
        raise FileExistsError(f"Output file already exists at {opt.output_file}")

    # Set status logging
    if experiment_spec.export.results_dir is not None:
        results_dir = experiment_spec.export.results_dir
    else:
        results_dir = os.path.join(experiment_spec.results_dir, "export")
        experiment_spec.export.results_dir = results_dir

    os.makedirs(results_dir, exist_ok=True)
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file,
                                                                 append=True))
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting OCRNet export"
    )

    export(opt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        run_experiment(experiment_spec=cfg)
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


if __name__ == '__main__':
    main()
