# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""ResNet backbone."""

from timm.models.resnet import BasicBlock, Bottleneck
from timm.models.resnet import ResNet as TimmResNet

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


class ResNet(TimmResNet, BackboneBase):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net.

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(self, *args, **kwargs):
        """Initialize the ResNet model.

        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
            activation_checkpoint (bool): Whether to use activation checkpointing. Default: `False`.
            freeze_at (list): List of keys corresponding to the stages or layers to freeze. If `None`, no specific
                layers are frozen. If `"all"`, the entire model is frozen and set to eval mode. Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the backbone will be frozen. Default: `False`.
        """
        in_chans = kwargs.get("in_chans", 3)
        num_classes = kwargs.get("num_classes", 1000)
        activation_checkpoint = kwargs.pop("activation_checkpoint", False)
        freeze_at = kwargs.pop("freeze_at", None)
        freeze_norm = kwargs.pop("freeze_norm", False)
        self.out_indices = kwargs.pop("out_indices", None)

        super().__init__(*args, **kwargs)  # TimmResNet initialization.
        BackboneBase.__init__(
            self,
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )

    def get_stage_dict(self):
        """Get the stage dictionary."""
        return {
            0: self.conv1,
            1: self.layer1,
            2: self.layer2,
            3: self.layer3,
            4: self.layer4,
        }

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head."""
        x = super().forward_features(x)
        x = super().forward_head(x, pre_logits=True)
        return x

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        outs = {}
        for i, name in enumerate(layer_names):
            x = getattr(self, name)(x)  # won't work with torchscript, but keeps code reasonable, FML
            if i in self.out_indices:
                outs[f"p{i}"] = x
        return outs

    def forward(self, x):
        """Forward."""
        x = self.forward_pre_logits(x)
        x = self.fc(x)
        return x


@BACKBONE_REGISTRY.register()
def resnet_18(**kwargs):
    """Constructs a ResNet-18 model."""
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_18d(**kwargs):
    """Constructs a ResNet-18D model."""
    return ResNet(BasicBlock, layers=[2, 2, 2, 2], stem_width=32, stem_type="deep", avg_down=True, **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_34(**kwargs):
    """Constructs a ResNet-34 model."""
    return ResNet(BasicBlock, layers=[3, 4, 6, 3], **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_34d(**kwargs):
    """Constructs a ResNet-34D model."""
    return ResNet(BasicBlock, layers=[3, 4, 6, 3], stem_width=32, stem_type="deep", avg_down=True, **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_50(**kwargs):
    """Constructs a ResNet-50 model."""
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_50d(**kwargs):
    """Constructs a ResNet-50D model."""
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type="deep", avg_down=True, **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_101(**kwargs):
    """Constructs a ResNet-101 model."""
    return ResNet(Bottleneck, layers=[3, 4, 23, 3], **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_101d(**kwargs):
    """Constructs a ResNet-101D model."""
    return ResNet(Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type="deep", avg_down=True, **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_152(**kwargs):
    """Constructs a ResNet-152 model."""
    return ResNet(Bottleneck, layers=[3, 8, 36, 3], **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_152d(**kwargs):
    """Constructs a ResNet-152D model."""
    return ResNet(Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type="deep", avg_down=True, **kwargs)
