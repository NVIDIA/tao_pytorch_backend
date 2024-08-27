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

"""Model utils of OCRNet"""

import argparse
import torch
import torch.nn.init as init

from nvidia_tao_pytorch.cv.ocrnet.model.model import Model
from nvidia_tao_pytorch.cv.ocrnet.utils.utils import load_checkpoint


def translate_model_config(experiment_spec):
    """Translate the model config to match with CLOVA"""
    parser = argparse.ArgumentParser()
    opt, _ = parser.parse_known_args()

    dataset_config = experiment_spec.dataset
    model_config = experiment_spec.model
    opt.batch_max_length = dataset_config.max_label_length
    opt.imgH = model_config.input_height
    opt.imgW = model_config.input_width
    opt.input_channel = model_config.input_channel

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

    if model_config.quantize:
        opt.quantize = True
    else:
        opt.quantize = False

    return opt


def create_quantize_model(model):
    """Add quantized module to the model.

    Args:
        model (torch.nn.Module): The model to which the quantized module is to be added.
    """
    from nvidia_tao_pytorch.cv.ocrnet.model.feature_extraction import BasicBlock
    from pytorch_quantization import nn as quant_nn

    # construct module dict
    modules_dict = {}
    for name, m in model.named_modules():
        modules_dict[name] = m
    for name, m in model.named_modules():
        if isinstance(m, BasicBlock):
            m.quantize = True
            m.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
        elif isinstance(m, torch.nn.modules.conv.Conv2d):
            bias = m.bias
            weight = m.weight
            # if bias:
            #     use_bias = True
            # else:
            #     use_bias = False
            use_bias = bool(bias is not None)
            quant_m = quant_nn.QuantConv2d(m.in_channels,
                                           m.out_channels,
                                           m.kernel_size,
                                           stride=m.stride,
                                           padding=m.padding,
                                           dilation=m.dilation,
                                           groups=m.groups,
                                           bias=use_bias,
                                           padding_mode=m.padding_mode)
            quant_m.weight = weight
            if use_bias:
                quant_m.bias = bias
            # Get the parent module name and attribute name
            parent_name = ".".join(name.split(".")[:-1])
            parent_module = modules_dict[parent_name]
            attribute_name = name.split(".")[-1]
            if attribute_name.isdigit():
                # process sequential module
                parent_module[int(attribute_name)] = quant_m
            else:
                setattr(parent_module, attribute_name, quant_m)


def load_for_finetune(model, state_dict):
    """Load the state_dict for finetune.

    Args:
        model (torch.nn.Module): The model to which the state_dict is to be loaded.
        state_dict (dict): A dictionary containing the state_dict to be loaded.
    """
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            if "initial_prob" in str(e):
                # ignore the constant variable in Attention:
                state_dict.pop("Prediction.initial_prob")
                state_dict.pop("Prediction.one_hot")
            # ignore the prediction layer weights when finetune
            if "weight" in str(e):
                if "generator" in str(e):
                    state_dict.pop("Prediction.generator.weight")
                    state_dict.pop("Prediction.generator.bias")
                    state_dict.pop("Prediction.attention_cell.rnn.weight_ih_l0")
                else:
                    state_dict.pop("Prediction.weight")
                    state_dict.pop("Prediction.bias")
        else:
            raise e


def build_ocrnet_model(experiment_spec, num_class):
    """Build OCRNet model of nn.module.

    Args:
        experiment_spec (dict): A dictionary of experiment specifications.
        num_class (int): The number of classes.

    Returns:
        nn.Module: The OCRNet model.
    """
    opt = translate_model_config(experiment_spec)
    opt.num_class = num_class

    # Init the stuff for QDQ
    if opt.quantize:
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization.tensor_quant import QuantDescriptor
        quant_desc_input = QuantDescriptor(calib_method='histogram')
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        from pytorch_quantization import quant_modules
        quant_modules.initialize()

    model = Model(opt)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # Load pretrained weights or resume model
    if experiment_spec.train.resume_training_checkpoint_path:
        model_path = experiment_spec.train.resume_training_checkpoint_path
        load_graph = True
        finetune = False
    elif experiment_spec.model.pruned_graph_path:
        model_path = experiment_spec.model.pruned_graph_path
        load_graph = True
        finetune = False
    elif experiment_spec.train.pretrained_model_path:
        model_path = experiment_spec.train.pretrained_model_path
        load_graph = False
        finetune = True
    elif experiment_spec.model.quantize_model_path:
        model_path = experiment_spec.model.quantize_model_path
        load_graph = True
        finetune = False
    else:
        model_path = None
        load_graph = False
        finetune = False

    if model_path:
        print(f'loading pretrained model from {model_path}')
        ckpt = load_checkpoint(model_path,
                               key=experiment_spec.encryption_key,
                               to_cpu=True)
        if not isinstance(ckpt, Model):
            if opt.FeatureExtraction == "FAN_tiny_2X":
                # For loading FAN_tiny imagenet pretrained
                ckpt = ckpt["state_dict"]
                new_ckpt = {}
                for key, val in ckpt.items():
                    if "patch_embed.backbone.stem.0" in key:
                        # For loading FAN_tiny model with input_channel==1
                        if opt.input_channel == 1:
                            new_ckpt[key] = val
                            continue
                    new_key = "FeatureExtraction." + ".".join(key.split(".")[1:])
                    new_ckpt[new_key] = val
                ckpt = new_ckpt
            else:
                # The public state_dict are with DP module
                ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}

            state_dict = ckpt
            if finetune:
                load_for_finetune(model, state_dict)
            else:
                model.load_state_dict(state_dict, strict=True)
        else:
            # The TAO OCRNet are without DP module
            if load_graph:
                model = ckpt
                if opt.quantize:
                    create_quantize_model(model)
            else:
                state_dict = ckpt.state_dict()
                load_for_finetune(model, state_dict)

    # Default to training mode
    model.train()
    return model
