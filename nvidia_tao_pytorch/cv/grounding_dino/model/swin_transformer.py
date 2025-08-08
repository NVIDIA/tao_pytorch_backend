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

""" Swin Transformer backbone. """

from nvidia_tao_pytorch.cv.backbone_v2.swin import SwinTransformer


def swin_tiny_224_1k(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """Swin-Tiny-224-IN1K model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
    """
    model = SwinTransformer(embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            pretrain_img_size=224,
                            out_indices=out_indices,
                            activation_checkpoint=activation_checkpoint,
                            num_classes=0,
                            **kwargs)
    model.num_features = [int(model.embed_dim * 2**i) for i in range(model.num_layers)]
    return model


def swin_base_224_22k(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """Swin-Base-224-IN22K model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
    """
    model = SwinTransformer(embed_dim=128,
                            depths=[2, 2, 12, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=7,
                            pretrain_img_size=224,
                            out_indices=out_indices,
                            activation_checkpoint=activation_checkpoint,
                            num_classes=0,
                            **kwargs)
    model.num_features = [int(model.embed_dim * 2**i) for i in range(model.num_layers)]
    return model


def swin_base_384_22k(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """Swin-Base-384-IN22K model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
    """
    model = SwinTransformer(embed_dim=128,
                            depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=12,
                            pretrain_img_size=384,
                            out_indices=out_indices,
                            activation_checkpoint=activation_checkpoint,
                            num_classes=0,
                            **kwargs)
    model.num_features = [int(model.embed_dim * 2**i) for i in range(model.num_layers)]
    return model


def swin_large_224_22k(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """Swin-Large-224-IN22K model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
    """
    model = SwinTransformer(embed_dim=192,
                            depths=[2, 2, 18, 2],
                            num_heads=[6, 12, 24, 48],
                            window_size=7,
                            pretrain_img_size=224,
                            out_indices=out_indices,
                            activation_checkpoint=activation_checkpoint,
                            num_classes=0,
                            **kwargs)
    model.num_features = [int(model.embed_dim * 2**i) for i in range(model.num_layers)]
    return model


def swin_large_384_22k(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """Swin-Large-384-IN22K model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
    """
    model = SwinTransformer(embed_dim=192,
                            depths=[2, 2, 18, 2],
                            num_heads=[6, 12, 24, 48],
                            window_size=12,
                            pretrain_img_size=384,
                            out_indices=out_indices,
                            activation_checkpoint=activation_checkpoint,
                            num_classes=0,
                            **kwargs)
    return model


swin_model_dict = {
    'swin_tiny_224_1k': swin_tiny_224_1k,
    'swin_tiny_patch4_window7_224': swin_tiny_224_1k,
    'swin_base_224_22k': swin_base_224_22k,
    'swin_base_patch4_window7_224': swin_base_224_22k,
    'swin_base_384_22k': swin_base_384_22k,
    'swin_base_patch4_window12_384': swin_base_384_22k,
    'swin_large_224_22k': swin_large_224_22k,
    'swin_large_patch4_window7_224': swin_large_224_22k,
    'swin_large_384_22k': swin_large_384_22k,
    'swin_large_patch4_window12_384': swin_large_384_22k,
}
