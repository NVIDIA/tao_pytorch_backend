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


"""FoundationStereo Refinement Module"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.extractor import EdgeNextConvEncoder


class FlowHead(nn.Module):
    """Flow Head"""

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """forward method for flowhead."""
        return self.conv2(self.relu(self.conv1(x)))


class DispHead(nn.Module):
    """Disparity head upscaler"""

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
            EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
            nn.Conv2d(input_dim, output_dim, 3, padding=1))

    def forward(self, x):
        """Forward method for DispHead."""
        return self.conv(x)


class ConvGRU(nn.Module):
    """
    Implements a Gated Recurrent Unit (GRU) with convolutional layers, designed for processing
    spatial data like feature maps in a neural network. This variant includes additional
    input parameters `cz`, `cr`, and `cq` which act as learnable biases or external
    inputs for each gate. This provides more flexibility than a standard ConvGRU by
    allowing the gates to be modulated by auxiliary information.

    The ConvGRU's core function is to update a hidden state tensor `h` based on a new
    input tensor `x`. It uses three gates: an update gate (`z`), a reset gate (`r`),
    and a candidate hidden state gate (`q`). These gates are computed using
    convolutional layers, enabling the unit to capture spatial dependencies.

    Args:
        hidden_dim (int): The number of channels for the hidden state `h` and the
                          output of the GRU.
        input_dim (int): The number of channels for the input tensor `x`.
        kernel_size (int, optional): The size of the convolutional kernel used for
                                     the gates. Defaults to 3.
    """

    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()

        # Convolutional layer for the update gate. It determines how much of the
        # previous hidden state to carry forward.
        self.convz = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

        # Convolutional layer for the reset gate. It controls how much of the previous
        # hidden state to forget.
        self.convr = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

        # Convolutional layer for the candidate hidden state. It computes a new
        # potential hidden state.
        self.convq = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, cz, cr, cq, *x_list):
        """
        Defines the forward pass of the ConvGRU.

        This method updates the hidden state `h` based on the input `x` and the
        learnable gate biases `cz`, `cr`, and `cq`.

        Args:
            h (torch.Tensor): The previous hidden state tensor with shape
                              `(batch_size, hidden_dim, H, W)`.
            cz (torch.Tensor): A bias tensor for the update gate, with shape
                               `(batch_size, hidden_dim, H, W)`.
            cr (torch.Tensor): A bias tensor for the reset gate, with shape
                               `(batch_size, hidden_dim, H, W)`.
            cq (torch.Tensor): A bias tensor for the candidate state gate, with shape
                               `(batch_size, hidden_dim, H, W)`.
            *x_list (list[torch.Tensor]): A list of one or more input tensors to be
                                          concatenated along the channel dimension.
                                          The combined tensor will have `input_dim` channels.

        Returns:
            torch.Tensor: The updated hidden state `h` with shape
                          `(batch_size, hidden_dim, H, W)`.
        """
        # Concatenate all input tensors in `x_list` along the channel dimension.
        x = torch.cat(x_list, dim=1)

        # Concatenate the previous hidden state `h` and the new input `x` for gate
        # computations.
        hx = torch.cat([h, x], dim=1)

        # Compute the update gate `z`. The bias `cz` is added to the convolutional
        # output before the sigmoid activation.
        z = torch.sigmoid(self.convz(hx) + cz)

        # Compute the reset gate `r`. The bias `cr` is added to the convolutional
        # output before the sigmoid activation.
        r = torch.sigmoid(self.convr(hx) + cr)

        # Compute the candidate hidden state `q`. The previous hidden state `h` is
        # first multiplied by the reset gate `r`. The result is concatenated with
        # the input `x` and passed through a convolution. The bias `cq` is added
        # before the tanh activation.
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)

        # Update the hidden state `h` using a linear interpolation between the
        # previous state `h` and the new candidate state `q`, weighted by the
        # update gate `z`.
        h = (1 - z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    """
    Implements a Basic Motion Encoder.

    This encoder takes a correlation volume and a disparity map as input, processes them
    through a series of convolutional layers, and concatenates the results to produce a
    compact feature representation. The goal is to distill the motion or depth information
    from these inputs into a feature tensor that can be used by a subsequent decoder
    or refinement network.

    Args:
        corr_levels:
        corr_radius:
                       which are used to calculate the input channels for the
                       correlation volume.
        ngroup (int, optional): The number of correlation groups used to compute
                                the correlation volume. Defaults to 8.
    """

    def __init__(self, corr_levels, corr_radius, ngroup=8):
        super(BasicMotionEncoder, self).__init__()

        cor_planes = corr_levels * (2 * corr_radius + 1) * (ngroup + 1)

        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 256, 3, padding=1)

        # Convolutional layers for processing the disparity map.
        # These layers increase the channel count and extract features from the disparity map.
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)

        # Final convolutional layer to combine features from both the correlation
        # volume and the disparity map. The output channels are set to a specific
        # size (128-1) to produce a feature map.
        self.conv = nn.Conv2d(64 + 256, 128 - 1, 3, padding=1)

    def forward(self, disp, corr):
        """
        Defines the forward pass of the BasicMotionEncoder.

        Args:
            disp (torch.Tensor): A tensor representing the input disparity map
                                 with shape `(batch_size, 1, height, width)`.
            corr (torch.Tensor): A tensor representing the input correlation volume
                                 with shape `(batch_size, cor_planes, height, width)`.

        Returns:
            torch.Tensor: A concatenated tensor containing the processed features
                          and the original disparity map. The output shape is
                          `(batch_size, 128, height, width)`.
        """
        # Process the correlation volume through its dedicated convolutional layers
        # and apply ReLU activation functions.
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))

        # Process the disparity map through its dedicated convolutional layers
        # and apply ReLU activation functions.
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        # Concatenate the processed correlation and disparity features along the channel dimension.
        cor_disp = torch.cat([cor, disp_], dim=1)

        # Pass the concatenated features through the final convolutional layer and apply ReLU.
        out = F.relu(self.conv(cor_disp))

        # Concatenate the output features with the original disparity map
        # to preserve the initial disparity information.
        return torch.cat([out, disp], dim=1)


def pool2x(x):
    """Average 2x pool helper function"""
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def pool4x(x):
    """Average 4x pool helper function"""
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x, dest):
    """Returns interpolation with the following specs."""
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class BasicMultiUpdateBlock(nn.Module):
    """
    Implements a multi-scale update block often found in recurrent networks for tasks like
    optical flow or disparity estimation.

    This block uses multiple Gated Recurrent Units (GRUs) to process and refine features
    at different scales. It integrates information from a motion encoder and propagates
    updates across scales, from coarse to fine, to iteratively improve the final prediction.
    The structure is designed to handle multiple feature resolutions
    (e.g., 1/4, 1/8, 1/16 of the original image size).

    Args:
        args (object): An object containing network configuration parameters.
                       Expected attributes include `n_gru_layers` and `n_downsample`.
        hidden_dims (list[int]): A list specifying the number of channels for the hidden
                                 states at each scale (e.g., `[128, 96, 64]` for scales
                                 1/16, 1/8, and 1/4, respectively).
        ngroup (int, optional): The number of correlation groups to pass to the
                                `BasicMotionEncoder`. Defaults to 8.
    """

    def __init__(self, args, hidden_dims=[], ngroup=8):
        super().__init__()
        self.args = args

        # Initialize the motion encoder to process correlation volumes and disparity maps.
        self.encoder = BasicMotionEncoder(args.corr_levels, args.corr_radius, ngroup=ngroup)
        encoder_output_dim = 128
        self.hidden_dims = hidden_dims

        # Initialize GRU cells for each scale. These GRUs update the hidden state
        # at a particular resolution based on its own input, its previous state,
        # and information from other scales.
        self.gru04 = ConvGRU(
            hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(
            hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(
            hidden_dims[0], hidden_dims[1])

        # Head for predicting the disparity delta from the finest-scale hidden state.
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)

        # A small network to generate mask features from the finest-scale hidden state.
        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr=None, disp=None,
                iter04=True, iter08=True, iter16=True, update=True):
        """
        Defines the forward pass for one recurrent update iteration.

        The forward pass updates the hidden states at different scales (1/16, 1/8, and 1/4)
        sequentially, starting from the coarsest scale. Each GRU receives the
        hidden state from a coarser scale (if available) and the feature map from the
        finer scale to incorporate context from both ends. Finally, it predicts
        a disparity update and a mask from the finest-scale hidden state.

        Args:
            net (list[torch.Tensor]): A list of hidden state tensors at different scales
                                      (e.g., `[hidden_state_1/4, hidden_state_1/8, hidden_state_1/16]`).
            inp (list[torch.Tensor]): A list of input feature tensors for each scale.
            corr (torch.Tensor, optional): The correlation volume. Required for the 1/4 scale update.
                                           Defaults to None.
            disp (torch.Tensor, optional): The current disparity map. Required for the 1/4 scale update.
                                           Defaults to None.
            iter04 (bool, optional): Whether to perform the update for the 1/4 scale. Defaults to True.
            iter08 (bool, optional): Whether to perform the update for the 1/8 scale. Defaults to True.
            iter16 (bool, optional): Whether to perform the update for the 1/16 scale. Defaults to True.
            update (bool, optional): If True, computes the disparity delta and mask features.
                                     If False, only updates the hidden states and returns them.
                                     Defaults to True.

        Returns:
            A tuple containing:
                - net (list[torch.Tensor]): The updated list of hidden state tensors.
                - mask_feat_4 (torch.Tensor): A feature map for the prediction mask.
                - delta_disp (torch.Tensor): The predicted disparity update.
        """
        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp


class RaftConvGRU(nn.Module):
    """
    Implements a Convolutional Gated Recurrent Unit (ConvGRU) as used in the
    RAFT (Recurrent All-Pairs Field Transforms)
    architecture for optical flow estimation.

    A ConvGRU is a variant of a standard GRU where the linear operations are replaced with convolutional
    layers.
    This allows the GRU to process spatial feature maps directly, making it suitable for tasks involving
    image or video data. It maintains a hidden state that is updated at each time step based on the current
    input and the previous hidden state, effectively propagating information through a sequence.

    Args:
        hidden_dim (int, optional): The number of channels in the hidden state and output. Defaults to 128.
        input_dim (int, optional): The number of channels in the input feature map. Defaults to 256.
        kernel_size (int, optional): The size of the convolutional kernel used for all internal convolutions.
        Defaults to 3.
    """

    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
        super().__init__()

        # The 'update gate' convolution. It determines how much of the new information
        # to incorporate into the hidden state.
        self.convz = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

        # The 'reset gate' convolution. It determines how much of the previous hidden state
        # to forget.
        self.convr = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

        # The 'candidate hidden state' convolution. It computes a new candidate for the
        # hidden state based on the reset gate's output and the current input.
        self.convq = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, x, hx):
        """
        Defines the forward pass of the ConvGRU.

        Args:
            h (torch.Tensor): The previous hidden state with shape `(batch_size, hidden_dim, H, W)`.
            x (torch.Tensor): The current input tensor with shape `(batch_size, input_dim, H, W)`.
            hx (torch.Tensor): The concatenation of the previous hidden state and the current input,
                               `torch.cat([h, x], dim=1)`, with shape
                               `(batch_size, hidden_dim + input_dim, H, W)`.
                               This is pre-concatenated to reduce redundant operations.

        Returns:
            torch.Tensor: The updated hidden state `h` with shape `(batch_size, hidden_dim, H, W)`.
        """
        # Calculate the update gate 'z' and the reset gate 'r' using sigmoid activation.
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))

        # Calculate the new candidate hidden state 'q' by first resetting the previous
        # hidden state using 'r', concatenating it with the input 'x', and then
        # passing it through the final convolution and tanh activation.
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        # Update the hidden state 'h' by combining the previous state 'h' and the
        # candidate state 'q' using the update gate 'z'.
        h = (1 - z) * h + z * q

        return h


class SelectiveConvGRU(nn.Module):
    """
    Implements a Selective Convolutional Gated Recurrent Unit (ConvGRU), which is a variation of a
    standard ConvGRU designed to selectively apply different GRU behaviors based on an attention mechanism.

    This module uses two different ConvGRUs internally: one with a small kernel size (e.g., 1x1) and
    another with a large kernel size (e.g., 3x3). It computes a new hidden state by taking a weighted
    average of the outputs from these two GRUs. The weights are determined by an attention map,
    allowing the model to dynamically choose between local (small kernel) and more contextual (large kernel)
    updates for different regions of the feature map.

    Args:
        hidden_dim (int, optional): The number of channels for the hidden state. Defaults to 128.
        input_dim (int, optional): The number of channels for the input feature map. Defaults to 256.
        small_kernel_size (int, optional): The kernel size for the small-kernel ConvGRU. Defaults to 1.
        large_kernel_size (int, optional): The kernel size for the large-kernel ConvGRU. Defaults to 3.
        patch_size (int, optional): This argument is not used in the current implementation but is
                                    included for potential future use or compatibility. Defaults to None.
    """

    def __init__(self, hidden_dim=128,
                 input_dim=256, small_kernel_size=1,
                 large_kernel_size=3, patch_size=None):
        super(SelectiveConvGRU, self).__init__()

        # Initial convolutional block to process the input feature map.
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # A second convolutional block to process the concatenated input and hidden state.
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, input_dim + hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # The small-kernel ConvGRU for learning local features.
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)

        # The large-kernel ConvGRU for learning contextual features.
        self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, *x):
        """
        Defines the forward pass of the SelectiveConvGRU.

        This method first processes the input features and concatenates them with the hidden state.
        It then computes the output of both the small-kernel and large-kernel GRUs.
        Finally, it uses the attention map to create a weighted combination of these two outputs,
        allowing the model to adaptively select the appropriate kernel size for each spatial location.

        Args:
            att (torch.Tensor): An attention map with shape `(batch_size, hidden_dim, H, W)`.
                                This map determines the weighting between the small and large GRU outputs.
                                Values are typically between 0 and 1.
            h (torch.Tensor): The previous hidden state with shape `(batch_size, hidden_dim, H, W)`.
            *x (list[torch.Tensor]): A list of one or more input tensors to be concatenated along
                                     the channel dimension. The final concatenated input will have
                                     `input_dim` channels.

        Returns:
            torch.Tensor: The updated hidden state `h`, which is a weighted combination of the outputs
                          from the two internal ConvGRUs. The shape is `(batch_size, hidden_dim, H, W)`.
        """
        # Concatenate multiple input tensors along the channel dimension.
        x = torch.cat(x, dim=1)

        # Process the concatenated input with the first convolutional block.
        x = self.conv0(x)

        # Concatenate the processed input and the previous hidden state.
        hx = torch.cat([x, h], dim=1)

        # Process the concatenated input-hidden state with the second convolutional block.
        hx = self.conv1(hx)

        # Compute the outputs of both the small and large GRUs and combine them using
        # the attention map 'att'. The attention map 'att' is applied to the small GRU's output,
        # and `(1 - att)` is applied to the large GRU's output, creating a weighted average.
        h = self.small_gru(h, x, hx) * att + self.large_gru(h, x, hx) * (1 - att)

        return h


class BasicSelectiveMultiUpdateBlock(nn.Module):
    """
    Implements a multi-scale update block with a selective recurrent
    mechanism for disparity refinement.

    This block is designed for recurrent networks, processing feature maps at multiple resolutions
    to iteratively refine a disparity prediction. It leverages `SelectiveConvGRU` units at each scale,
    which can dynamically adjust their receptive fields (via a small or large kernel) based on an
    attention map. This allows the model to perform fine-grained updates in homogeneous areas and
    use a wider context near object boundaries or occlusions. The architecture processes features
    from coarse to fine, propagating information between scales to produce a refined disparity delta
    and a confidence mask.

    Args:
        args (object): An object containing network configuration parameters.
                       Expected attributes include `n_gru_layers`, which determines the number
                       of scales at which updates are performed.
        hidden_dim (int, optional): The number of channels for the hidden state and GRU outputs at
                                    each scale. Defaults to 128.
        volume_dim (int, optional): The number of correlation groups to pass to the
                                    `BasicMotionEncoder`. Defaults to 8.
    """

    def __init__(self, args, hidden_dim=128, volume_dim=8):
        super().__init__()
        self.args = args

        # Initializes a motion encoder to process correlation volumes and disparity maps.
        self.encoder = BasicMotionEncoder(args.corr_levels, args.corr_radius, volume_dim)

        # Initializes SelectiveConvGRU units for each scale based on the number of GRU layers.
        if args.n_gru_layers == 3:
            self.gru16 = SelectiveConvGRU(hidden_dim, hidden_dim * 2)
        if args.n_gru_layers >= 2:
            self.gru08 = SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers == 3) + hidden_dim * 2)
        self.gru04 = SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers > 1) + hidden_dim * 2)

        # Head for predicting the disparity delta from the finest-scale hidden state.
        self.disp_head = DispHead(hidden_dim, 256)

        # A small network to generate a mask from the finest-scale hidden state.
        self.mask = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr, disp, att):
        """
        Defines the forward pass for one recurrent update iteration.

        The forward pass updates the hidden states from the coarsest scale (1/16) to the finest (1/4).
        Each `SelectiveConvGRU` receives its respective attention map, hidden state, input features,
        and information from adjacent scales (either pooled from a finer scale or interpolated
        from a coarser one).
        The final outputs are a refined list of hidden states, a confidence mask, and a disparity delta.

        Args:
            net (list[torch.Tensor]): A list of hidden state tensors at different scales
                                      (e.g., `[hidden_state_1/4, hidden_state_1/8, hidden_state_1/16]`).
            inp (list[torch.Tensor]): A list of input feature tensors for each scale.
            corr (torch.Tensor): The correlation volume.
            disp (torch.Tensor): The current disparity map.
            att (list[torch.Tensor]): A list of attention maps corresponding
                                      to each scale, used by the `SelectiveConvGRU`
                                      to modulate its behavior

        Returns:
            A tuple containing:
                - net (list[torch.Tensor]): The updated list of hidden state tensors.
                - mask (torch.Tensor): A feature map representing a confidence mask.
                - delta_disp (torch.Tensor): The predicted disparity update.
        """
        # Update hidden state at 1/16 scale (if 3 GRU layers are used).
        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(att[2], net[2], inp[2], pool2x(net[1]))

        # Update hidden state at 1/8 scale (if 2 or more GRU layers are used).
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                # Update with information from both 1/4 and 1/16 scales.
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]), interp(net[2], net[1]))
            else:
                # Update with information from only the 1/4 scale.
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]))

        # Process correlation and disparity to get motion features.
        motion_features = self.encoder(disp, corr)
        motion_features = torch.cat([inp[0], motion_features], dim=1)

        # Update hidden state at 1/4 scale.
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(att[0], net[0], motion_features, interp(net[1], net[0]))

        # Predict the disparity delta from the finest-scale hidden state.
        delta_disp = self.disp_head(net[0])

        # Generate and scale the mask from the finest-scale hidden state.
        # The scaling factor is used to balance gradients during training.
        mask = .25 * self.mask(net[0])
        return net, mask, delta_disp
