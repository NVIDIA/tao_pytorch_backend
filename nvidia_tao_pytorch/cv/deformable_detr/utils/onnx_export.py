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

""" Generates TRT compatible DDETR onnx model. """

import torch
from torch.onnx import register_custom_op_symbolic

import onnx
import numpy as np

import onnx_graphsurgeon as gs


# register plugin
def nvidia_msda(g, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
    """Returns nvidia_msda."""
    return g.op("nvidia::MultiscaleDeformableAttnPlugin_TRT", value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)


class ONNXExporter(object):
    """Onnx Exporter."""

    @classmethod
    def setUpClass(cls):
        """SetUpclass to set the manual seed for reproduceability"""
        torch.manual_seed(123)

    def export_model(self, model, batch_size, onnx_file, dummy_input, do_constant_folding=False, opset_version=12,
                     output_names=None, input_names=None, verbose=False):
        """ Export_model.

        The do_constant_folding = False avoids MultiscaleDeformableAttnPlugin_TRT error (tensors on 2 devices) when torch > 1.9.0.
        However, it would cause tensorrt 8.0.3.4 (nvcr.io/nvidia/pytorch:21.11-py3 env) reports clip node error.
        This error is fixed in tensorrt >= 8.2.1.8 (nvcr.io/nvidia/tensorrt:22.01-py3).

        Args:
            model (nn.Module): torch model to export.
            batch_size (int): batch size of the ONNX model. -1 means dynamic batch size.
            onnx_file (str): output path of the onnx file.
            dummy_input (torch.Tensor): input tensor.
            do_constant_folding (bool): flag to indicate whether to fold constants in the ONNX model.
            opset_version (int): opset_version of the ONNX file.
            output_names (str): output names of the ONNX file.
            input_names (str): input names of the ONNX file.
            verbose (bool): verbosity level.
        """
        if batch_size is None or batch_size == -1:
            dynamic_axes = {"inputs": {0: "batch"}, "pred_logits": {0: "batch"}, "pred_boxes": {0: "batch"}}
        else:
            dynamic_axes = None

        # CPU version requires opset_version > 16
        if not next(model.parameters()).is_cuda and opset_version < 16:
            print(f"CPU version of Deformable MHA requires opset version larger than 16. Overriding provided opset {opset_version} to 16.")
            opset_version = 16

        register_custom_op_symbolic('nvidia::MultiscaleDeformableAttnPlugin_TRT', nvidia_msda, opset_version)

        torch.onnx.export(model, dummy_input, onnx_file,
                          input_names=input_names, output_names=output_names, export_params=True,
                          training=torch.onnx.TrainingMode.EVAL, opset_version=opset_version, do_constant_folding=do_constant_folding,
                          custom_opsets={"nvidia": 1}, verbose=verbose, dynamic_axes=dynamic_axes)

    @staticmethod
    def check_onnx(onnx_file):
        """Check onnx file.

        Args:
            onnx_file (str): path to ONNX file.
        """
        model = onnx.load(onnx_file)
        onnx.checker.check_model(model)

    @staticmethod
    def onnx_change(onnx_file):
        """Make ddetr onnx compatible with TRT. Additionally, fold constants.

        Args:
            onnx_file (str): path to ONNX file.
        """
        graph = gs.import_onnx(onnx.load(onnx_file))
        for node in graph.nodes:
            if node.op == "MultiscaleDeformableAttnPlugin_TRT":
                node.attrs = {"name": "MultiscaleDeformableAttnPlugin_TRT", "version": "1", "namespace": ""}
                new_inputs = []
                for i, inp in enumerate(node.inputs):
                    if i in (1, 2) and hasattr(inp, "values"):
                        new_inp = gs.Constant(name=inp.name, values=inp.values.astype(np.int32))
                        new_inputs.append(new_inp)
                    else:
                        new_inputs.append(inp)
                node.inputs = new_inputs

        # Setting constant folding in torch result in error due to some layers still in CPU
        # Limit workspace size to 1GB to disable folding for MatMul
        graph.fold_constants(size_threshold=1024 * 1024 * 1024)
        graph.cleanup().toposort()
        onnx.save(gs.export_onnx(graph), onnx_file)
