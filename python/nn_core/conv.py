"""
Convolutional Neural Network Layers
====================================

This module provides convolutional layers for processing spatial data (images, sequences).
All modules take Tensor inputs and return Tensor outputs, with gradients computed
automatically via the computational graph.

Modules:
- Conv1d: 1D convolution for sequences
- Conv2d: 2D convolution for images
- DepthwiseConv2d: Per-channel spatial filtering
- PointwiseConv2d: 1x1 convolution for channel mixing
- DepthwiseSeparableConv2d: Efficient factorized convolution
- DilatedConv2d: Atrous convolution for large receptive fields
- TransposedConv2d: Upsampling via transposed convolution
- ASPPBlock: Atrous Spatial Pyramid Pooling

Functional classes for autograd:
- Conv1dFunction, Conv2dFunction, ConvTranspose2dFunction

Helper functions:
- im2col_2d, col2im_2d: Efficient convolution via matrix multiplication
- calculate_output_shape, calculate_receptive_field

References
----------
- CS231n: Convolutional Networks (https://cs231n.github.io/convolutional-networks/)
- "ImageNet Classification with Deep CNNs" (AlexNet)
- "MobileNets: Efficient CNNs" (Depthwise Separable)
- "Multi-Scale Context Aggregation by Dilated Convolutions" (Atrous)
"""

# Implementation Status: STUBS
# Complexity: Hard
# Prerequisites: foundations/computational_graph, foundations/functionals

import numpy as np
from typing import Tuple, Union, Optional, List

from .module import Module, Parameter
from python.foundations import Tensor, convert_to_function
from . import conv_functional


# =============================================================================
# 1D Convolution
# =============================================================================

class Conv1d(Module):
    """
    1D Convolution Layer for Sequential Data.

    Performs convolution operation on 1D input sequences. Useful for processing
    temporal data, audio signals, or sequential embeddings.

    Forward:
        y[n, c_out, l] = Σ_{c_in} Σ_{k} x[n, c_in, l*stride + k] * w[c_out, c_in, k] + b[c_out]

    Output length:
        L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1

    Example:
        >>> conv = Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        >>> x = Tensor(np.random.randn(8, 16, 100))  # Batch of 8 sequences
        >>> y = conv(x)
        >>> y.shape  # (8, 32, 100)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True
    ):
        super().__init__()

        if kernel_size <= 0 or stride <= 0 or dilation <= 0:
            raise ValueError("kernel_size, stride, and dilation must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize weights (Kaiming initialization)
        weight_data = np.random.randn(out_channels, in_channels, kernel_size) * np.sqrt(2.0 / (kernel_size * in_channels))
        self.weight = Parameter(weight_data)

        if bias:
            self.bias = Parameter(np.zeros(out_channels))
        else:
            self.bias = None
        self.conv1d = convert_to_function(conv_functional.Conv1d)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform 1D convolution.

        Args:
            x: Input Tensor (batch_size, in_channels, seq_length)

        Returns:
            Output Tensor (batch_size, out_channels, output_length)
        """
        return self.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}"
        )


# =============================================================================
# 2D Convolution
# =============================================================================

class Conv2d(Module):
    """
    2D Convolution Layer - Core Building Block for CNNs.

    Applies learned filters across spatial dimensions (height, width) to extract
    hierarchical features from images and feature maps.

    Shape Convention (NCHW - PyTorch format):
        Input: (batch_size, in_channels, height, width)
        Weight: (out_channels, in_channels/groups, kernel_h, kernel_w)
        Output: (batch_size, out_channels, height_out, width_out)

    Output size:
        H_out = floor((H_in + 2*pad_h - dilation_h*(K_h - 1) - 1) / stride_h) + 1
        W_out = floor((W_in + 2*pad_w - dilation_w*(K_w - 1) - 1) / stride_w) + 1

    Example:
        >>> conv = Conv2d(3, 64, kernel_size=3, padding=1)
        >>> x = Tensor(np.random.randn(8, 3, 224, 224))  # ImageNet batch
        >>> y = conv(x)
        >>> y.shape  # (8, 64, 224, 224)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, str, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if isinstance(padding, int):
            padding = (padding, padding)

        if out_channels % groups != 0 or in_channels % groups != 0:
            raise ValueError(
                f"groups ({groups}) must divide both in_channels ({in_channels}) "
                f"and out_channels ({out_channels})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if isinstance(padding, str) else padding
        self.dilation = dilation
        self.groups = groups

        # Kaiming initialization
        k_h, k_w = kernel_size
        weight_data = np.random.randn(out_channels, in_channels // groups, k_h, k_w) * np.sqrt(2.0 / (k_h * k_w * in_channels // groups))
        self.weight = Parameter(weight_data)

        if bias:
            self.bias = Parameter(np.zeros(out_channels))
        else:
            self.bias = None

        self.conv2d = convert_to_function(conv_functional.Conv2d)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform 2D convolution.

        Args:
            x: Input Tensor (batch_size, in_channels, height, width)

        Returns:
            Output Tensor (batch_size, out_channels, height_out, width_out)
        """
        return self.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}"
        )


# =============================================================================
# Depthwise Separable Convolution Components
# =============================================================================

class DepthwiseConv2d(Module):
    """
    Depthwise 2D Convolution - Spatial Filtering Per Channel.

    Performs separate convolutions for each input channel with its own kernel.
    Equivalent to grouped convolution with groups=in_channels.

    Parameters: in_channels × K_h × K_w (much fewer than standard conv!)

    Example:
        >>> dw_conv = DepthwiseConv2d(64, kernel_size=3, padding=1)
        >>> x = Tensor(np.random.randn(8, 64, 56, 56))
        >>> y = dw_conv(x)  # Same spatial dims, same channels
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Weight shape: (in_channels, 1, kernel_h, kernel_w)
        k_h, k_w = kernel_size
        weight_data = np.random.randn(in_channels, 1, k_h, k_w) * np.sqrt(2.0 / (k_h * k_w))
        self.weight = Parameter(weight_data)

        if bias:
            self.bias = Parameter(np.zeros(in_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        """Perform depthwise convolution (groups=in_channels)."""
        raise NotImplementedError(
            "TODO: Implement DepthwiseConv2d forward\n"
            "Hint: Use Conv2d with groups=in_channels, or\n"
            "process each channel independently with its own kernel."
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )


class PointwiseConv2d(Module):
    """
    Pointwise 2D Convolution - Channel Mixing via 1×1 Convolution.

    Uses 1×1 kernels to mix information across channels without changing
    spatial dimensions. Ultra-efficient for channel expansion/compression.

    Parameters: in_channels × out_channels (no spatial parameters!)

    Example:
        >>> pw_conv = PointwiseConv2d(64, 128)
        >>> x = Tensor(np.random.randn(8, 64, 56, 56))
        >>> y = pw_conv(x)  # (8, 128, 56, 56) - channels expanded
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Weight shape: (out_channels, in_channels, 1, 1)
        weight_data = np.random.randn(out_channels, in_channels, 1, 1) * np.sqrt(2.0 / in_channels)
        self.weight = Parameter(weight_data)

        if bias:
            self.bias = Parameter(np.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        """Perform pointwise (1×1) convolution."""
        raise NotImplementedError(
            "TODO: Implement PointwiseConv2d forward\n"
            "Hint: 1×1 conv is just matrix multiply per spatial position:\n"
            "  output[b,:,i,j] = weight @ input[b,:,i,j] + bias\n"
            "Or reshape: (B×H×W, in_c) @ (in_c, out_c) → (B×H×W, out_c)"
        )

    def extra_repr(self) -> str:
        return f"{self.in_channels}, {self.out_channels}"


class DepthwiseSeparableConv2d(Module):
    """
    Depthwise Separable 2D Convolution - Efficient Factorization.

    Combines depthwise and pointwise convolutions for ~7-9× parameter reduction.
    Core building block of MobileNet, ShuffleNet, Xception.

    Two-stage process:
        1. Depthwise: in_c → in_c (spatial features per channel)
        2. Pointwise: in_c → out_c (channel mixing)

    Parameter comparison (in_c=32, out_c=64, kernel=3×3):
        Standard Conv2d: 32 × 64 × 3 × 3 = 18,432 params
        DepthwiseSeparable: (32×9) + (32×64) = 2,336 params  (~7.9× reduction!)

    Example:
        >>> dw_sep = DepthwiseSeparableConv2d(32, 64, kernel_size=3, padding=1)
        >>> x = Tensor(np.random.randn(8, 32, 56, 56))
        >>> y = dw_sep(x)  # (8, 64, 56, 56)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Two stages
        self.depthwise = DepthwiseConv2d(
            in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias
        )
        self.pointwise = PointwiseConv2d(in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform depthwise separable convolution.

        Args:
            x: Input Tensor (batch_size, in_channels, height, width)

        Returns:
            Output Tensor (batch_size, out_channels, height_out, width_out)
        """
        raise NotImplementedError(
            "TODO: Implement DepthwiseSeparableConv2d forward\n"
            "Hint:\n"
            "  # Stage 1: Depthwise - per-channel spatial filtering\n"
            "  x_dw = self.depthwise(x)  # (B, in_c, H', W')\n"
            "  \n"
            "  # Stage 2: Pointwise - channel mixing\n"
            "  output = self.pointwise(x_dw)  # (B, out_c, H', W')\n"
            "  \n"
            "  return output"
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )


# =============================================================================
# Dilated (Atrous) Convolution
# =============================================================================

class DilatedConv2d(Module):
    """
    Dilated 2D Convolution - Exponential Receptive Field Expansion.

    Inserts spaces between kernel elements, expanding receptive field without
    adding parameters. Essential for semantic segmentation (DeepLab) and
    audio processing (WaveNet).

    Effective kernel size: K_eff = K + (K-1)*(dilation-1)

    Examples:
        - 3×3 kernel, dilation=1: RF = 3×3
        - 3×3 kernel, dilation=2: RF = 5×5 (same parameters!)
        - 3×3 kernel, dilation=4: RF = 9×9

    Example:
        >>> dilated = DilatedConv2d(64, 128, kernel_size=3, dilation=2, padding=2)
        >>> x = Tensor(np.random.randn(8, 64, 56, 56))
        >>> y = dilated(x)  # Same spatial dims, expanded receptive field
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        k_h, k_w = kernel_size
        weight_data = np.random.randn(out_channels, in_channels // groups, k_h, k_w) * np.sqrt(2.0 / (k_h * k_w * in_channels // groups))
        self.weight = Parameter(weight_data)

        if bias:
            self.bias = Parameter(np.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        """Perform dilated (atrous) 2D convolution."""
        raise NotImplementedError(
            "TODO: Implement DilatedConv2d forward\n"
            "Hint: Same as Conv2d but kernel samples at dilation spacing:\n"
            "  input_pos = oh*stride + kh*dilation - padding\n"
            "  (Note: kh*dilation instead of just kh)"
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}"
        )


class ASPPBlock(Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) - Multi-Scale Context Module.

    Used in DeepLab for semantic segmentation. Applies parallel dilated
    convolutions with different rates to capture multi-scale context.

    Architecture:
        Branch 1: Conv(1×1, d=1) - local detail
        Branch 2: Conv(3×3, d=6) - medium context
        Branch 3: Conv(3×3, d=12) - large context
        Branch 4: Conv(3×3, d=18) - largest context
        Branch 5: GlobalAvgPool → 1×1 → Upsample - global context
        → Concatenate → 1×1 conv → output

    Example:
        >>> aspp = ASPPBlock(256, 256, dilations=[1, 6, 12, 18])
        >>> x = Tensor(np.random.randn(8, 256, 64, 64))
        >>> y = aspp(x)  # Multi-scale context aggregated
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: List[int] = None
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 6, 12, 18]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations
        self.num_branches = len(dilations) + 1  # +1 for image-level branch

    def forward(self, x: Tensor) -> Tensor:
        """Apply ASPP multi-scale context aggregation."""
        raise NotImplementedError(
            "TODO: Implement ASPPBlock forward\n"
            "Hint:\n"
            "  1. Create parallel branches with different dilations\n"
            "  2. Add global average pooling branch\n"
            "  3. Concatenate all branches\n"
            "  4. Final 1×1 conv for projection"
        )

    def extra_repr(self) -> str:
        return f"{self.in_channels}, {self.out_channels}, dilations={self.dilations}"

# =============================================================================
# 3D Convolution
# =============================================================================

class Conv3d(Module):
    """
    3D Convolution for volumetric data (video, 3D images).

    Input shape: (batch, in_channels, depth, height, width)
    Output shape: (batch, out_channels, depth_out, height_out, width_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        k_d, k_h, k_w = kernel_size
        weight_data = np.random.randn(out_channels, in_channels // groups, k_d, k_h, k_w) * np.sqrt(2.0 / (k_d * k_h * k_w * in_channels // groups))
        self.weight = Parameter(weight_data)

        if bias:
            self.bias = Parameter(np.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("TODO: Implement Conv3d forward")


# =============================================================================
# Transposed Convolutions (Deconvolution)
# =============================================================================

class ConvTranspose1d(Module):
    """
    1D Transposed Convolution for upsampling sequences.

    Optimized implementation using GEMM + col2im (reuses Conv1d's C kernels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        weight_data = np.random.randn(in_channels, out_channels // groups, kernel_size) * np.sqrt(2.0 / (kernel_size * in_channels))
        self.weight = Parameter(weight_data)

        if bias:
            self.bias = Parameter(np.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.conv_transpose1d = convert_to_function(conv_functional.ConvTranspose1d)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_transpose1d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding,
            self.dilation, self.groups,
        )


class ConvTranspose2d(Module):
    """
    2D Transposed Convolution for upsampling images.

    Optimized implementation using GEMM + col2im (reuses Conv2d's C kernels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        k_h, k_w = kernel_size
        weight_data = np.random.randn(in_channels, out_channels // groups, k_h, k_w) * np.sqrt(2.0 / (k_h * k_w * in_channels))
        self.weight = Parameter(weight_data)

        if bias:
            self.bias = Parameter(np.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.conv_transpose2d = convert_to_function(conv_functional.ConvTranspose2d)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_transpose2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding,
            self.dilation, self.groups,
        )


class ConvTranspose3d(Module):
    """3D Transposed Convolution for upsampling volumes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        k_d, k_h, k_w = kernel_size
        weight_data = np.random.randn(in_channels, out_channels // groups, k_d, k_h, k_w) * np.sqrt(2.0 / (k_d * k_h * k_w * in_channels))
        self.weight = Parameter(weight_data)

        if bias:
            self.bias = Parameter(np.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("TODO: Implement ConvTranspose3d forward")


# Alias for compatibility
SeparableConv2d = DepthwiseSeparableConv2d
