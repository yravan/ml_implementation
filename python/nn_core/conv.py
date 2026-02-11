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
from python.foundations import Tensor
from python.foundations.functionals import Function


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
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform 1D convolution.

        Args:
            x: Input Tensor (batch_size, in_channels, seq_length)

        Returns:
            Output Tensor (batch_size, out_channels, output_length)
        """
        raise NotImplementedError(
            "TODO: Implement Conv1d forward\n"
            "Hint: Use Conv1dFunction.apply(x, self.weight, self.bias, ...)\n"
            "Or use im2col approach:\n"
            "  1. Extract all patches from input using sliding window\n"
            "  2. Reshape to (batch*output_size, kernel_size*in_channels)\n"
            "  3. Perform matrix multiply: output = patches @ weight.T + bias\n"
            "  4. Reshape to (batch_size, out_channels, output_length)"
        )

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
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform 2D convolution.

        Args:
            x: Input Tensor (batch_size, in_channels, height, width)

        Returns:
            Output Tensor (batch_size, out_channels, height_out, width_out)
        """
        raise NotImplementedError(
            "TODO: Implement Conv2d forward\n"
            "Hint: Use Conv2dFunction.apply(x, self.weight, self.bias, ...)\n"
            "Or use im2col approach:\n"
            "  1. Extract all K_h × K_w patches from input via sliding window\n"
            "  2. Reshape to (batch*h_out*w_out, in_channels*k_h*k_w)\n"
            "  3. Perform matrix multiplication: output = patches @ weight.T + bias\n"
            "  4. Reshape to (batch_size, out_channels, h_out, w_out)"
        )

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
# Transposed Convolution
# =============================================================================

class TransposedConv2d(Module):
    """
    Transposed 2D Convolution - Upsampling Layer.

    Also known as deconvolution or fractionally-strided convolution.
    Used for upsampling in autoencoders, U-Net, and GANs.

    The forward pass is mathematically equivalent to the backward pass of
    Conv2d w.r.t. the input.

    Output shape:
        H_out = (H_in - 1)*stride_h - 2*padding_h + dilation_h*(K_h - 1) + output_padding_h + 1

    Example:
        >>> up_conv = TransposedConv2d(64, 32, kernel_size=4, stride=2, padding=1)
        >>> x = Tensor(np.random.randn(8, 64, 28, 28))
        >>> y = up_conv(x)  # (8, 32, 56, 56) - upsampled 2×
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

        # Note: Weight shape has channels swapped compared to Conv2d
        k_h, k_w = kernel_size
        weight_data = np.random.randn(in_channels, out_channels // groups, k_h, k_w) * np.sqrt(2.0 / (k_h * k_w * out_channels // groups))
        self.weight = Parameter(weight_data)

        if bias:
            self.bias = Parameter(np.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform transposed 2D convolution (upsampling).

        Args:
            x: Input Tensor (batch_size, in_channels, height, width)

        Returns:
            Upsampled Tensor (batch_size, out_channels, height_out, width_out)
        """
        raise NotImplementedError(
            "TODO: Implement TransposedConv2d forward\n"
            "Hint: Input dilation approach:\n"
            "  1. Dilate input: insert (stride-1) zeros between elements\n"
            "  2. Pad with kernel_size - 1 - original_padding\n"
            "  3. Perform regular convolution with flipped weights\n"
            "Or use ConvTranspose2dFunction.apply(...)"
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )


# =============================================================================
# Functional Classes for Autograd
# =============================================================================

class Conv1dFunction(Function):
    """1D Convolution functional operation for autograd."""

    def __init__(self, stride: int = 1, padding: int = 0, dilation: int = 1):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute 1D convolution."""
        raise NotImplementedError("TODO: Implement Conv1dFunction forward")

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute gradients for 1D convolution."""
        raise NotImplementedError("TODO: Implement Conv1dFunction backward")


class Conv2dFunction(Function):
    """2D Convolution functional operation for autograd."""

    def __init__(
        self,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1
    ):
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute 2D convolution using im2col."""
        raise NotImplementedError("TODO: Implement Conv2dFunction forward")

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute gradients for 2D convolution."""
        raise NotImplementedError("TODO: Implement Conv2dFunction backward")


class ConvTranspose2dFunction(Function):
    """2D Transposed Convolution functional operation for autograd."""

    def __init__(
        self,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1
    ):
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute transposed 2D convolution."""
        raise NotImplementedError("TODO: Implement ConvTranspose2dFunction forward")

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute gradients for transposed 2D convolution."""
        raise NotImplementedError("TODO: Implement ConvTranspose2dFunction backward")


# =============================================================================
# Helper Functions
# =============================================================================

def im2col_2d(
    x: np.ndarray,
    kernel_h: int,
    kernel_w: int,
    stride: Tuple[int, int],
    dilation: Tuple[int, int]
) -> np.ndarray:
    """
    Convert image patches to columns for efficient convolution.

    Takes sliding windows from input and arranges them as columns.
    This converts convolution into matrix multiplication.

    Args:
        x: Input tensor (batch, channels, height, width)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: (stride_h, stride_w)
        dilation: (dilation_h, dilation_w)

    Returns:
        cols: Column matrix (batch, channels*kernel_h*kernel_w, out_h*out_w)
    """
    raise NotImplementedError("TODO: Implement im2col_2d")


def col2im_2d(
    cols: np.ndarray,
    x_shape: Tuple[int, ...],
    kernel_h: int,
    kernel_w: int,
    stride: Tuple[int, int],
    dilation: Tuple[int, int]
) -> np.ndarray:
    """
    Convert columns back to image (inverse of im2col).

    Used in backward pass to convert gradient columns to gradient image.

    Args:
        cols: Column matrix (batch, channels*kernel_h*kernel_w, out_h*out_w)
        x_shape: Original input shape (batch, channels, height, width)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: (stride_h, stride_w)
        dilation: (dilation_h, dilation_w)

    Returns:
        x: Reconstructed tensor (batch, channels, height, width)
    """
    raise NotImplementedError("TODO: Implement col2im_2d")


def calculate_output_shape(
    input_shape: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1)
) -> Tuple[int, int]:
    """
    Calculate output spatial dimensions for 2D convolution.

    Formula:
        H_out = floor((H_in + 2*pad_h - dil_h*(K_h - 1) - 1) / stride_h) + 1
        W_out = floor((W_in + 2*pad_w - dil_w*(K_w - 1) - 1) / stride_w) + 1
    """
    h_in, w_in = input_shape
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation

    h_out = ((h_in + 2*p_h - d_h*(k_h - 1) - 1) // s_h) + 1
    w_out = ((w_in + 2*p_w - d_w*(k_w - 1) - 1) // s_w) + 1

    return (h_out, w_out)


def calculate_transposed_output_shape(
    input_shape: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    output_padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1)
) -> Tuple[int, int]:
    """
    Calculate output shape for transposed convolution.

    Formula:
        H_out = (H_in - 1)*stride_h - 2*pad_h + dil_h*(K_h - 1) + out_pad_h + 1
    """
    h_in, w_in = input_shape
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    op_h, op_w = output_padding
    d_h, d_w = dilation

    h_out = (h_in - 1)*s_h - 2*p_h + d_h*(k_h - 1) + op_h + 1
    w_out = (w_in - 1)*s_w - 2*p_w + d_w*(k_w - 1) + op_w + 1

    return (h_out, w_out)


def calculate_receptive_field(layer_configs: List[dict]) -> int:
    """
    Calculate cumulative receptive field for stacked conv layers.

    Args:
        layer_configs: List of dicts with keys: kernel_size, stride, dilation

    Returns:
        Total receptive field size
    """
    rf = 1
    for config in layer_configs:
        k = config.get('kernel_size', 3)
        s = config.get('stride', 1)
        d = config.get('dilation', 1)
        rf += (k - 1) * s * d
    return rf


def count_conv_parameters(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    groups: int = 1,
    bias: bool = True
) -> int:
    """Count learnable parameters in a Conv2d layer."""
    k_h, k_w = kernel_size
    num_weights = out_channels * (in_channels // groups) * k_h * k_w
    num_bias = out_channels if bias else 0
    return num_weights + num_bias


def count_depthwise_separable_parameters(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int] = (3, 3),
    bias: bool = True
) -> Tuple[int, int, float]:
    """
    Compare parameters: standard conv vs depthwise separable.

    Returns:
        (standard_params, depthwise_sep_params, reduction_factor)
    """
    k_h, k_w = kernel_size

    # Standard convolution
    standard = out_channels * in_channels * k_h * k_w
    if bias:
        standard += out_channels

    # Depthwise separable
    depthwise = in_channels * k_h * k_w
    if bias:
        depthwise += in_channels
    pointwise = in_channels * out_channels
    if bias:
        pointwise += out_channels
    depthwise_sep = depthwise + pointwise

    reduction = standard / depthwise_sep if depthwise_sep > 0 else 1.0

    return standard, depthwise_sep, reduction


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
    """1D Transposed Convolution for upsampling sequences."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
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
        self.groups = groups

        weight_data = np.random.randn(in_channels, out_channels // groups, kernel_size) * np.sqrt(2.0 / (kernel_size * in_channels))
        self.weight = Parameter(weight_data)

        if bias:
            self.bias = Parameter(np.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("TODO: Implement ConvTranspose1d forward")


class ConvTranspose2d(Module):
    """2D Transposed Convolution for upsampling images."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        k_h, k_w = kernel_size
        weight_data = np.random.randn(in_channels, out_channels // groups, k_h, k_w) * np.sqrt(2.0 / (k_h * k_w * in_channels))
        self.weight = Parameter(weight_data)

        if bias:
            self.bias = Parameter(np.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("TODO: Implement ConvTranspose2d forward")


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
