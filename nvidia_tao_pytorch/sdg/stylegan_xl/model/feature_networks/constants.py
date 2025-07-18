# Original source taken from https://github.com/autonomousvision/stylegan-xl
#
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

""" Various constants used for model selection and configuration. """

# List of models from the torchvision library
TORCHVISION = [
    "vgg11_bn",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "densenet121",
    "densenet169",
    "densenet201",
    "inception_v3",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "shufflenet_v2_x0_5",
    "mobilenet_v2",
    "wide_resnet50_2",
    "mnasnet0_5",
    "mnasnet1_0",
    "ghostnet_100",
    "cspresnet50",
    "fbnetc_100",
    "spnasnet_100",
    "resnet50d",
    "resnet26",
    "resnet26d",
    "seresnet50",
    "resnetblur50",
    "resnetrs50",
    "tf_mixnet_s",
    "tf_mixnet_m",
    "tf_mixnet_l",
    "ese_vovnet19b_dw",
    "ese_vovnet39b",
    "res2next50",
    "gernet_s",
    "gernet_m",
    "repvgg_a2",
    "repvgg_b0",
    "repvgg_b1",
    "repvgg_b1g4",
    "revnet",
    "dm_nfnet_f1",
    "nfnet_l0",
]

# List of RegNet models
REGNETS = [
    "regnetx_002",
    "regnetx_004",
    "regnetx_006",
    "regnetx_008",
    "regnetx_016",
    "regnetx_032",
    "regnetx_040",
    "regnetx_064",
    "regnety_002",
    "regnety_004",
    "regnety_006",
    "regnety_008",
    "regnety_016",
    "regnety_032",
    "regnety_040",
    "regnety_064",
]

# List of EfficientNet models for ImageNet
EFFNETS_IMAGENET = [
    'tf_efficientnet_b0',
    'tf_efficientnet_b1',
    'tf_efficientnet_b2',
    'tf_efficientnet_b3',
    'tf_efficientnet_b4',
    'tf_efficientnet_b0_ns',
]

# List of EfficientNet models for Inception
EFFNETS_INCEPTION = [
    'tf_efficientnet_lite0',
    'tf_efficientnet_lite1',
    'tf_efficientnet_lite2',
    'tf_efficientnet_lite3',
    'tf_efficientnet_lite4',
    'tf_efficientnetv2_b0',
    'tf_efficientnetv2_b1',
    'tf_efficientnetv2_b2',
    'tf_efficientnetv2_b3',
    'efficientnet_b1',
    'efficientnet_b1_pruned',
    'efficientnet_b2_pruned',
    'efficientnet_b3_pruned',
]

# Combined list of all EfficientNet models
EFFNETS = EFFNETS_IMAGENET + EFFNETS_INCEPTION

# List of Vision Transformer models for ImageNet
VITS_IMAGENET = [
    'deit_tiny_distilled_patch16_224',
    'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224',
]

# List of Vision Transformer models for Inception
VITS_INCEPTION = [
    'vit_base_patch16_224'
]

# Combined list of all Vision Transformer models
VITS = VITS_IMAGENET + VITS_INCEPTION

# List of CLIP models
CLIP = [
    'resnet50_clip'
]

# Combined list of all models
ALL_MODELS = TORCHVISION + REGNETS + EFFNETS + VITS + CLIP

# Group models according to input normalization

# Models normalized for ImageNet
NORMALIZED_IMAGENET = TORCHVISION + REGNETS + EFFNETS_IMAGENET + VITS_IMAGENET

# Models normalized for Inception
NORMALIZED_INCEPTION = EFFNETS_INCEPTION + VITS_INCEPTION

# Models normalized for CLIP
NORMALIZED_CLIP = CLIP
