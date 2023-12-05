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

""" Generates TRT compatible CenterPose onnx model. """

import torch
import onnx


class ONNXExporter(object):
    """Onnx Exporter"""

    @classmethod
    def setUpClass(cls):
        """SetUpclass to set the manual seed for reproduceability"""
        torch.manual_seed(123)

    def export_model(self, model, batch_size, onnx_file, dummy_input, do_constant_folding=False, opset_version=12,
                     output_names=None, input_names=None, verbose=False):
        """ Export the onnx model.
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
            dynamic_axes = {'input': {0: 'batch_size'}, 'bboxes': {0: 'batch_size'}, 'scores': {0: 'batch_size'},
                            'kps': {0: 'batch_size'}, 'clses': {0: 'batch_size'}, 'obj_scale': {0: 'batch_size'},
                            'kps_displacement_mean': {0: 'batch_size'}, 'kps_heatmap_mean': {0: 'batch_size'}}
        else:
            dynamic_axes = None

        # CPU version requires opset_version > 16
        if not next(model.parameters()).is_cuda and opset_version < 16:
            print(f"CPU version of Deformable MHA requires opset version larger than 16. Overriding provided opset {opset_version} to 16.")
            opset_version = 16

        torch.onnx.export(model, dummy_input, onnx_file,
                          input_names=input_names, output_names=output_names, export_params=True,
                          training=torch.onnx.TrainingMode.EVAL, opset_version=opset_version, do_constant_folding=do_constant_folding, verbose=verbose, dynamic_axes=dynamic_axes)

    @staticmethod
    def check_onnx(onnx_file):
        """Check onnx file.

        Args:
            onnx_file (str): path to ONNX file.
        """
        model = onnx.load(onnx_file)
        onnx.checker.check_model(model)
