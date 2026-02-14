"""
Deformable Convolution
======================

Deformable Convolution from "Deformable Convolutional Networks".
https://arxiv.org/abs/1703.06211

Deformable convolutions learn spatial offsets for sampling locations,
allowing the network to adapt its receptive field to object shape.
"""

import numpy as np
from typing import Tuple, Optional


def deform_conv2d(
    input: np.ndarray,
    offset: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Perform deformable convolution.

    For each output position, instead of sampling from a fixed grid,
    learnable 2D offsets are added to the regular grid positions.

    Standard conv samples at: p_n (fixed grid positions)
    Deform conv samples at: p_n + delta_p_n (with learned offsets)

    Optionally, modulation masks can weight the importance of each sample.

    Args:
        input: (N, C_in, H, W) input feature map
        offset: (N, 2*kH*kW, H_out, W_out) learnable offsets
                2 values (x, y offset) per kernel position
        weight: (C_out, C_in, kH, kW) convolution weights
        bias: (C_out,) optional bias
        stride: Convolution stride
        padding: Input padding
        dilation: Kernel dilation
        mask: (N, kH*kW, H_out, W_out) optional modulation mask
              Values in [0, 1] that weight each sampling position

    Returns:
        (N, C_out, H_out, W_out) output feature map
    """
    raise NotImplementedError("TODO: Implement deform_conv2d")


class DeformConv2d:
    """
    Deformable Convolution v2 layer.

    This module includes the offset and mask prediction convolutions.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Size of convolution kernel
        stride: Convolution stride
        padding: Input padding
        dilation: Kernel dilation
        groups: Number of groups for grouped convolution
        bias: Whether to include bias

    Example:
        >>> deform_conv = DeformConv2d(64, 128, kernel_size=3, padding=1)
        >>> output = deform_conv(input)  # offset/mask predicted internally
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Weight for main convolution
        self.weight = None  # (out_channels, in_channels // groups, kH, kW)
        self.bias = None if not bias else None  # (out_channels,)

        # Offset convolution predicts 2D offsets for each kernel position
        # For kH x kW kernel: 2 * kH * kW offset values
        self.offset_conv = None

        # Mask convolution predicts modulation mask (DCNv2)
        self.mask_conv = None

    def __call__(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement DeformConv2d")
