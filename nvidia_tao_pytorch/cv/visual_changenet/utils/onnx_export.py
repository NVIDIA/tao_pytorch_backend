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

""" Generates TRT compatible Visual ChangeNet onnx model. """

import torch
import onnx


class ONNXExporter(object):
    """Onnx Exporter"""

    @classmethod
    def setUpClass(cls):
        """SetUpclass"""
        torch.manual_seed(123)

    def export_model(self, model, batch_size, onnx_file, dummy_input, task, do_constant_folding=False, opset_version=12,
                     output_names=None, input_names=None, verbose=False):
        """ Export_model
        The do_constant_folding = False avoids MultiscaleDeformableAttnPlugin_TRT error (tensors on 2 devices) when torch > 1.9.0.
        However, it would cause tensorrt 8.0.3.4 (nvcr.io/nvidia/pytorch:21.11-py3 env) reports clip node error.
        This error is fixed in tensorrt >= 8.2.1.8 (nvcr.io/nvidia/tensorrt:22.01-py3).
        """
        if batch_size is None or batch_size == -1:
            # namings given to input and output layers in onnx exported model
            if task == 'segment':
                dynamic_axes = {'input0': {0: 'batch_size'},
                                'input1': {0: 'batch_size'},
                                'output0': {0: 'batch_size'},
                                'output1': {0: 'batch_size'},
                                'output2': {0: 'batch_size'},
                                'output3': {0: 'batch_size'},
                                'output_final': {0: 'batch_size'}}
            elif task == 'classify':
                dynamic_axes = {"input_1": {0: "batch"},
                                "input_2": {0: "batch"},
                                "output": {0: "batch"}
                                }
        else:
            dynamic_axes = None

        torch.onnx.export(model, dummy_input, onnx_file,
                          input_names=input_names, output_names=output_names, export_params=True,
                          training=torch.onnx.TrainingMode.EVAL, opset_version=opset_version, do_constant_folding=do_constant_folding,
                          verbose=verbose, dynamic_axes=dynamic_axes)

    @staticmethod
    def check_onnx(onnx_file):
        """Check_onnx"""
        model = onnx.load(onnx_file)
        onnx.checker.check_model(model)
