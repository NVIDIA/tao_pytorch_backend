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

"""OCRNet feature extraction module."""
import torch.nn as nn
import torch.nn.functional as F
from pytorch_quantization import nn as quant_nn


class VGG_FeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, input_channel, output_channel=512):
        """Init.

        Args:
            input_channel (int): The number of input channels.
            output_channel (int, optional): The number of output channels. Default is 512.
        """
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # 256x8x25
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))  # 512x1x24

    def forward(self, input):  # pylint: disable=redefined-builtin
        """Forward."""
        return self.ConvNet(input)


class RCNN_FeatureExtractor(nn.Module):
    """ FeatureExtractor of GRCNN (https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf) """

    def __init__(self, input_channel, output_channel=512):
        """Init.

        Args:
            input_channel (int): The number of input channels.
            output_channel (int, optional): The number of output channels. Default is 512.
        """
        super(RCNN_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64 x 16 x 50
            GRCL(self.output_channel[0], self.output_channel[0], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, 2),  # 64 x 8 x 25
            GRCL(self.output_channel[0], self.output_channel[1], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 128 x 4 x 26
            GRCL(self.output_channel[1], self.output_channel[2], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 256 x 2 x 27
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True))  # 512 x 1 x 26

    def forward(self, input):  # pylint: disable=redefined-builtin
        """Forward."""
        return self.ConvNet(input)


class ResNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel, output_channel=512, quantize=False, no_maxpool1=False):
        """Init.

        Args:
            input_channel (int): The number of input channels.
            output_channel (int, optional): The number of output channels. Default is 512.
            quantize (bool, optional): Whether to use quantization. Default is False.
        """
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3], quantize=quantize, no_maxpool1=no_maxpool1)

    def forward(self, input):  # pylint: disable=redefined-builtin
        """Forward."""
        return self.ConvNet(input)


# For Gated RCNN
class GRCL(nn.Module):
    """Gated RCNN."""

    def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
        """Init.

        Args:
            input_channel (int): The number of input channels.
            output_channel (int): The number of output channels.
            num_iteration (int): The number of iterations of recursion.
            kernel_size (int): The size of the kernel.
            pad (int): The amount of padding.
        """
        super(GRCL, self).__init__()
        self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
        self.wf_u = nn.Conv2d(input_channel, output_channel, kernel_size, 1, pad, bias=False)
        self.wr_x = nn.Conv2d(output_channel, output_channel, kernel_size, 1, pad, bias=False)

        self.BN_x_init = nn.BatchNorm2d(output_channel)

        self.num_iteration = num_iteration
        self.GRCL = [GRCL_unit(output_channel) for _ in range(num_iteration)]
        self.GRCL = nn.Sequential(*self.GRCL)

    def forward(self, input):  # pylint: disable=redefined-builtin
        """ The input of GRCL is consistant over time t, which is denoted by u(0)
        thus wgf_u / wf_u is also consistant over time t.
        """
        wgf_u = self.wgf_u(input)
        wf_u = self.wf_u(input)
        x = F.relu(self.BN_x_init(wf_u))

        for i in range(self.num_iteration):
            x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))

        return x


class GRCL_unit(nn.Module):
    """Gated RCNN unit."""

    def __init__(self, output_channel):
        """Init.
        Args:
            output_channel (int): The number of output channels.
        """
        super(GRCL_unit, self).__init__()
        self.BN_gfu = nn.BatchNorm2d(output_channel)
        self.BN_grx = nn.BatchNorm2d(output_channel)
        self.BN_fu = nn.BatchNorm2d(output_channel)
        self.BN_rx = nn.BatchNorm2d(output_channel)
        self.BN_Gx = nn.BatchNorm2d(output_channel)

    def forward(self, wgf_u, wgr_x, wf_u, wr_x):
        """Performs a forward pass through the GRCL_unit network."""
        G_first_term = self.BN_gfu(wgf_u)
        G_second_term = self.BN_grx(wgr_x)
        G = F.sigmoid(G_first_term + G_second_term)

        x_first_term = self.BN_fu(wf_u)
        x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
        x = F.relu(x_first_term + x_second_term)

        return x


class BasicBlock(nn.Module):
    """Basic Block for ResNet."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, quantize=False):
        """Init.

        Args:
            inplanes (int): The number of input channels.
            planes (int): The number of output channels.
            stride (int, optional): The stride of the convolutional layer. Default is 1.
            downsample (nn.Module, optional): The downsampling layer. Default is None.
            quantize (bool, optional): Whether to use quantization. Default is False.
        """
        super(BasicBlock, self).__init__()
        self.quantize = quantize
        self.conv1 = self._conv3x3(inplanes, planes, quantize=self.quantize)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes, quantize=self.quantize)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if self.quantize:
            self.residual_quantizer = \
                quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)

    def _conv3x3(self, in_planes, out_planes, stride=1, quantize=False):
        """3x3 convolution with padding"""
        if quantize:
            return quant_nn.QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                        padding=1, bias=False)

        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        """forward."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.quantize:
            out += self.residual_quantizer(residual)
        else:
            out += residual
        out = self.relu(out)

        return out


def get_conv2d(quantize=False):
    """Helper function for quantize model."""
    if quantize:
        return quant_nn.QuantConv2d
    return nn.Conv2d


class ResNet(nn.Module):
    """ResNet module."""

    def __init__(self, input_channel, output_channel, block, layers, quantize=False, no_maxpool1=False):
        """Init.

        Args:
            input_channel (int): The number of input channels.
            output_channel (int): The number of output channels.
            block (nn.Module): The block to use for the ResNet network.
            layers (list): A list of integers specifying the number of blocks in each layer.
            quantize (bool, optional): Whether to use quantization. Default is False.
        """
        super(ResNet, self).__init__()
        self.quantize = quantize
        self.no_maxpool1 = no_maxpool1
        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = get_conv2d(self.quantize)(input_channel, int(output_channel / 16),
                                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = get_conv2d(self.quantize)(int(output_channel / 16), self.inplanes,
                                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if not self.no_maxpool1:
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = get_conv2d(self.quantize)(self.output_channel_block[0], self.output_channel_block[
                                               0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = get_conv2d(self.quantize)(self.output_channel_block[1], self.output_channel_block[
                                               1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = get_conv2d(self.quantize)(self.output_channel_block[2], self.output_channel_block[
                                               2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = get_conv2d(self.quantize)(self.output_channel_block[3], self.output_channel_block[
                                                 3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = get_conv2d(self.quantize)(self.output_channel_block[3], self.output_channel_block[
                                                 3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _conv1x1(self, in_planes, out_planes, stride=1, quantize=False):
        """conv1x1 helper."""
        if quantize:
            return quant_nn.QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                        bias=False)

        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                         bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        """make resnet block."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )
            downsample = nn.Sequential(
                self._conv1x1(self.inplanes, planes * block.expansion,
                              stride=stride, quantize=self.quantize),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """forward."""
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        if not self.no_maxpool1:
            x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x
