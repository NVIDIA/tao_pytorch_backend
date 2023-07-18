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
#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
"""Export module."""
import os
import torch
from torch.onnx import register_custom_op_symbolic
import copy
import onnx
import onnx_graphsurgeon as onnx_gs
from torchvision.ops import DeformConv2d

import tempfile
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs

from nvidia_tao_pytorch.cv.ocdnet.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocdnet.model.pl_ocd_model import OCDnetModel
from omegaconf import OmegaConf


def symbolic_dcnv2_forward(g, *inputs):
    """symbolic_dcnv2_forward"""
    # weights as last input to align with TRT plugin
    return g.op("ModulatedDeformConv2d", inputs[0], inputs[2], inputs[3], inputs[1])


# Register custom symbolic function
register_custom_op_symbolic("torchvision::deform_conv2d", symbolic_dcnv2_forward, 11)


class Export():
    """Export OCDNet model."""

    def __init__(
        self, model_path, config_file,
        width, height, opset_version,
        gpu_id=0
    ):
        """Initialize."""
        self.model_path = model_path
        self.config_file = config_file
        self.opset_version = opset_version
        self.gpu_id = gpu_id
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
            checkpoint = {key.replace("model.", ""): value for key, value in checkpoint.items()}
        config = OmegaConf.to_container(config_file)
        config['model']['pretrained'] = False
        config["dataset"]["train_dataset"] = config["dataset"]["validate_dataset"]
        self.model = OCDnetModel(config)
        layers = checkpoint.keys()
        ckpt = dict()
        # Support loading official pretrained weights for eval
        for layer in layers:
            new_layer = layer
            if new_layer.startswith("model.module."):
                new_layer = new_layer[13:]
            if new_layer == "decoder.in5.weight":
                new_layer = "neck.in5.weight"
            elif new_layer == "decoder.in4.weight":
                new_layer = "neck.in4.weight"
            elif new_layer == "decoder.in3.weight":
                new_layer = "neck.in3.weight"
            elif new_layer == "decoder.in2.weight":
                new_layer = "neck.in2.weight"
            elif new_layer == "decoder.out5.0.weight":
                new_layer = "neck.out5.0.weight"
            elif new_layer == "decoder.out4.0.weight":
                new_layer = "neck.out4.0.weight"
            elif new_layer == "decoder.out3.0.weight":
                new_layer = "neck.out3.0.weight"
            elif new_layer == "decoder.out2.weight":
                new_layer = "neck.out2.weight"
            elif new_layer == "decoder.binarize.0.weight":
                new_layer = "head.binarize.0.weight"
            elif new_layer == "decoder.binarize.1.weight":
                new_layer = "head.binarize.1.weight"
            elif new_layer == "decoder.binarize.1.bias":
                new_layer = "head.binarize.1.bias"
            elif new_layer == "decoder.binarize.1.running_mean":
                new_layer = "head.binarize.1.running_mean"
            elif new_layer == "decoder.binarize.1.running_var":
                new_layer = "head.binarize.1.running_var"
            elif new_layer == "decoder.binarize.3.weight":
                new_layer = "head.binarize.3.weight"
            elif new_layer == "decoder.binarize.3.bias":
                new_layer = "head.binarize.3.bias"
            elif new_layer == "decoder.binarize.4.weight":
                new_layer = "head.binarize.4.weight"
            elif new_layer == "decoder.binarize.4.bias":
                new_layer = "head.binarize.4.bias"
            elif new_layer == "decoder.binarize.4.running_mean":
                new_layer = "head.binarize.4.running_mean"
            elif new_layer == "decoder.binarize.4.running_var":
                new_layer = "head.binarize.4.running_var"
            elif new_layer == "decoder.binarize.6.weight":
                new_layer = "head.binarize.6.weight"
            elif new_layer == "decoder.binarize.6.bias":
                new_layer = "head.binarize.6.bias"
            elif new_layer == "decoder.thresh.0.weight":
                new_layer = "head.thresh.0.weight"
            elif new_layer == "decoder.thresh.1.weight":
                new_layer = "head.thresh.1.weight"
            elif new_layer == "decoder.thresh.1.bias":
                new_layer = "head.thresh.1.bias"
            elif new_layer == "decoder.thresh.1.running_mean":
                new_layer = "head.thresh.1.running_mean"
            elif new_layer == "decoder.thresh.1.running_var":
                new_layer = "head.thresh.1.running_var"
            elif new_layer == "decoder.thresh.3.weight":
                new_layer = "head.thresh.3.weight"
            elif new_layer == "decoder.thresh.3.bias":
                new_layer = "head.thresh.3.bias"
            elif new_layer == "decoder.thresh.4.weight":
                new_layer = "head.thresh.4.weight"
            elif new_layer == "decoder.thresh.4.bias":
                new_layer = "head.thresh.4.bias"
            elif new_layer == "decoder.thresh.4.running_mean":
                new_layer = "head.thresh.4.running_mean"
            elif new_layer == "decoder.thresh.4.running_var":
                new_layer = "head.thresh.4.running_var"
            elif new_layer == "decoder.thresh.6.weight":
                new_layer = "head.thresh.6.weight"
            elif new_layer == "decoder.thresh.6.bias":
                new_layer = "head.thresh.6.bias"
            elif "num_batches_tracked" in new_layer:
                continue
            elif "backbone.fc" in new_layer:
                continue
            elif "backbone.smooth" in new_layer:
                continue
            ckpt[new_layer] = checkpoint[layer]

        self.model.model.load_state_dict(ckpt)
        self.model.to(self.device)

    def export(self):
        """Export."""
        self.model.eval()
        dummy_image = torch.zeros(
            (1, 3, 544, 960),
            dtype=torch.float32,
            device='cuda:0'
        )

        if self.config_file.export.results_dir is not None:
            results_dir = self.config_file.export.results_dir
        else:
            results_dir = os.path.join(self.config_file.results_dir, "export")
            self.config_file.export.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

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
            message="Starting OCDNet export"
        )

        # Set default output filename if the filename isn't provided over the command line.
        if self.config_file.export.onnx_file is None:
            split_name = os.path.splitext(self.model_path)[0]
            self.config_file.export.onnx_file = "{}.onnx".format(split_name)

        # Warn the user if an exported file already exists.
        if os.path.exists(self.config_file.export.onnx_file):
            raise FileExistsError(f"Output file already exists at {self.config_file.export.onnx_file}")

        self.output_model = self.config_file.export.onnx_file

        handle, temp_onnx = tempfile.mkstemp()
        os.close(handle)

        torch.onnx.export(
            self.model,
            (dummy_image,),
            temp_onnx,
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            keep_initializers_as_inputs=True,
            input_names=['input'],
            output_names=['pred'],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
            }
        )
        # Import and add DCNv2 attributes
        onnx_model = onnx.load(temp_onnx)
        gs_graph = onnx_gs.import_onnx(onnx_model)
        layer_dict = {}
        attrs_dict = {}
        for name, layer in self.model.named_modules():
            if isinstance(layer, DeformConv2d):
                attrs_dict["stride"] = list(layer.stride)
                attrs_dict["padding"] = list(layer.padding)
                attrs_dict["dilation"] = list(layer.dilation)
                attrs_dict["group"] = 1
                attrs_dict["deformable_group"] = 1
                name = name.replace("model.backbone.", "") + ".ModulatedDeformConv2d"
                layer_dict[name] = copy.deepcopy(attrs_dict)
        for node in gs_graph.nodes:
            if node.op == "ModulatedDeformConv2d":
                key = (".".join(node.name.split("/")[-3:]))
                node.attrs = layer_dict[key]

        gs_graph.fold_constants()
        gs_graph.cleanup()
        new_onnx_model = onnx_gs.export_onnx(gs_graph)

        onnx.save(new_onnx_model, self.output_model)

        print("Model exported to {}".format(self.output_model))

        os.remove(temp_onnx)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="export", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the export process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)

    try:
        exporter = Export(
            config_file=cfg,
            model_path=cfg.export.checkpoint,
            width=cfg.export.width,
            height=cfg.export.height,
            opset_version=cfg.export.opset_version
        )
        exporter.export()
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


if __name__ == "__main__":
    main()
