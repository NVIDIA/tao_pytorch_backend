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
# flake8: noqa: I1101

import onnx
from typing import Optional
import torch
from torch.onnx import _constants, _type_utils, symbolic_helper
from torch.onnx._internal import _beartype, jit_utils
from torch.onnx.symbolic_opset12 import _onnx_symbolic

import numpy as np
from functools import partial
import onnxruntime as rt

from mmcv.onnx import register_extra_symbolics
from nvidia_tao_pytorch.cv.classification.models.dinov2_vit import DinoV2ViT, interpolate_pos_encoding


@_onnx_symbolic("aten::triu")
@_beartype.beartype
def triu(g: jit_utils.GraphContext, self, diagonal, out=None):
    """
    Triu operator.
    """
    return g.op("Trilu", self, diagonal, upper_i=1)

# Ported from
# https://github.com/microsoft/onnxscript/blob/6b1b81700b4523f31d8c6d3321e5d8ef5d42b764/onnxscript/function_libs/torch_aten/ops/nn.py#L1504
# aten_scaled_dot_product_attention
# NOTE: Need op.Trilu
@_onnx_symbolic("aten::_scaled_dot_product_attention")
@symbolic_helper.parse_args("v", "v", "v", "v", "f", "b", "v")
@_beartype.beartype
def scaled_dot_product_attention(
    g: jit_utils.GraphContext,
    query: torch._C.Value, # noqa pylint: disable=I1101
    key: torch._C.Value, # noqa pylint: disable=I1101
    value: torch._C.Value, # noqa pylint: disable=I1101
    attn_mask: Optional[torch._C.Value] = None, # noqa pylint: disable=I1101
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[torch._C.Value] = None, # noqa pylint: disable=I1101
):
    """
    Scale Dot-Product Attention operator.
    """
    assert (not is_causal) or (
        is_causal and symbolic_helper._is_none(attn_mask)
    ), "is_causal and attn_mask cannot be set at the same time"

    scale = symbolic_helper._maybe_get_const(scale, "f")
    if scale == 0 or symbolic_helper._is_none(scale):
        scale = _attention_scale(g, query)

    if is_causal:
        attn_mask = _causal_attention_mask(g, query, key)

    # Swap the last two axes of key
    # NOTE: onnx-script has different logic here, because the attribute perms in
    # transpose needs list of ints
    key_shape_builtin = symbolic_helper._get_tensor_rank(key)
    key_transposed_axes = list(range(key_shape_builtin))
    key_transposed_axes[-1], key_transposed_axes[-2] = (
        key_transposed_axes[-2],
        key_transposed_axes[-1],
    )
    key_transposed = g.op("Transpose", key, perm_i=key_transposed_axes)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = g.op("Mul", query, g.op("Sqrt", scale))
    key_transposed_scaled = g.op("Mul", key_transposed, g.op("Sqrt", scale))
    mul_qk = g.op("MatMul", query_scaled, key_transposed_scaled)

    if symbolic_helper._is_none(attn_mask):
        mul_qk_add = mul_qk
    elif (
        _type_utils.JitScalarType.from_value(attn_mask)
        == _type_utils.JitScalarType.BOOL
    ):
        # Turn the Boolean mask to float: attn_mask.masked_fill(not attn_mask, -float('inf'))
        const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
        const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
        attn_mask = g.op("Where", attn_mask, const_zero, const_neg_inf)
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    elif (
        _type_utils.JitScalarType.from_value(attn_mask)
        == _type_utils.JitScalarType.FLOAT
    ):
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    else:
        raise ValueError(
            f"Unsupported type for attn_mask: {_type_utils.JitScalarType.from_value(attn_mask)}"
        )

    attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)

    if dropout_p != 0:
        attn_weight = g.op(
            "Dropout",
            attn_weight,
            g.op("Constant", value_t=torch.tensor(dropout_p, dtype=torch.float)),
        )

    return g.op("MatMul", attn_weight, value), attn_weight


@_beartype.beartype
def _attention_scale(
    g: jit_utils.GraphContext, query: torch._C.Value # noqa pylint: disable=I1101
) -> torch._C.Value: # noqa pylint: disable=I1101
    """Calculate the scale factor for the attention result.

    Args:
        query: Tensor of shape [..., L, E]

    Returns:
        Scalar scale factor := 1 / math.sqrt(query.size(-1))
    """
    query_shape = g.op("Shape", query)
    query_shape_last = g.op(
        "Slice",
        query_shape,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)),
        g.op(
            "Constant", value_t=torch.tensor([_constants.INT64_MAX], dtype=torch.int64)
        ),
    )
    embedding_size = g.op(
        "Cast",
        query_shape_last,
        to_i=_type_utils.JitScalarType.from_value(query).onnx_type(),
    )
    const_one = g.op("Constant", value_t=torch.tensor([1.0], dtype=torch.float))
    scale = g.op("Div", const_one, g.op("Sqrt", embedding_size))
    return scale


@_beartype.beartype
def _causal_attention_mask(
    g: jit_utils.GraphContext, query: torch._C.Value, key: torch._C.Value # noqa pylint: disable=I1101
) -> torch._C.Value: # noqa pylint: disable=I1101
    """Create a causal mask for the given query and key tensors.

    Equivalent to::
        mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_mask = torch.zeros(L, S, dtype=torch.float)
        attn_mask = attn_mask.masked_fill(not mask, -float('inf'))

    Args:
        query: Tensor of shape [..., L, E]
        key: Tensor of shape [..., S, E]

    Returns:
        Tensor of shape [L, S]
    """
    query_shape = g.op("Shape", query)
    key_shape = g.op("Shape", key)

    last_idx = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
    second_last_idx = g.op("Constant", value_t=torch.tensor([-2], dtype=torch.int64))
    target_length = g.op("Slice", query_shape, second_last_idx, last_idx)
    source_length = g.op("Slice", key_shape, second_last_idx, last_idx)
    # attn_mask = torch.ones(L, S) := {
    size = g.op("Concat", target_length, source_length, axis_i=0)
    const_one = g.op("Constant", value_t=torch.tensor([1.0]))
    attn_mask = g.op("Expand", const_one, size)
    # }
    attn_mask = g.op("Trilu", attn_mask, upper_i=0)
    # The causal mask has 0s in the lower triangle and -inf in the upper triangle.
    const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
    const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
    attn_mask = g.op(
        "Where", g.op("Equal", attn_mask, const_zero), const_neg_inf, const_zero
    )
    return attn_mask


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

    if isinstance(model.backbone, DinoV2ViT):
        reshaped_embed = interpolate_pos_encoding(
            model.backbone.pos_embed.data,
            input_shape[2],
            input_shape[3],
        )
        setattr(model.backbone, "pos_embed", torch.nn.Parameter(reshaped_embed))

        if logger:
            logger.info("Reshape positional encoding for DinoV2ViT")

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

        np.testing.assert_allclose(pytorch_result, onnx_result, rtol=1e-04, atol=1e-06)

        if logger:
            logger.info('The outputs are same between Pytorch and ONNX')
