# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Original source taken from https://github.com/open-mmlab/mmclassification

# Copyright 2019 OpenMMLAB

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ONNX related tools."""

import onnx
import torch

import numpy as np
from functools import partial
import onnxruntime as rt

from mmcv.onnx import register_extra_symbolics


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of classification classes
    """
    (N, _, _, _) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_labels = rng.randint(
        low=0, high=num_classes, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs


def pytorch_to_onnx(model,
                    input_shape,
                    num_classes,
                    opset_version=11,
                    show=False,
                    output_file='tmp.onnx',
                    verify=False,
                    logger=None):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    model.cpu().eval()
    if hasattr(model.head, 'num_classes'):
        num_classes = model.head.num_classes
    # Some backbones use `num_classes=-1` to disable top classifier.
    elif getattr(model.backbone, 'num_classes', -1) > 0:
        num_classes = model.backbone.num_classes
    else:
        raise AttributeError('Cannot find "num_classes" in both head and '
                             'backbone, please check the config file.')

    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_list = [img[None, :] for img in imgs]

    # replace original forward function
    origin_forward_test = model.forward_test
    model.forward = partial(model.forward_test)
    register_extra_symbolics(opset_version)

    # support dynamic shape export
    dynamic_axes = {
        'input_1': {
            0: 'batch',
        },
        'probs': {
            0: 'batch'
        }
    }

    with torch.no_grad():
        torch.onnx.export(
            model, (img_list, ),
            output_file,
            input_names=['input_1'],
            output_names=['probs'],
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            verbose=show,
            opset_version=opset_version)

    if logger:
        logger.info(f'Successfully exported ONNX model: {output_file}')

    model.forward_test = origin_forward_test
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # test the dynamic model
        dynamic_test_inputs = _demo_mm_inputs(
            (input_shape[0], input_shape[1], input_shape[2],
                input_shape[3]), model.head.num_classes)
        imgs = dynamic_test_inputs.pop('imgs')
        img_list = [img[None, :] for img in imgs]

        # check the numerical value
        # get pytorch output
        pytorch_result = model(img_list)[0]

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1), "The input dimension is not equal to one"
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: img_list[0].detach().numpy()})[0][0]
        if not np.testing.assert_allclose(pytorch_result, onnx_result, rtol=1e-04):
            raise ValueError(
                'The outputs are different between Pytorch and ONNX')
        if logger:
            logger.info('The outputs are same between Pytorch and ONNX')
