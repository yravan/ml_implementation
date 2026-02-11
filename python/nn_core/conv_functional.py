"""
Convolutional Functional Operations
====================================

This module provides functional operations for convolution layers.
Function classes handle the forward/backward computation with np.ndarray,
while Module classes in conv.py wrap these for Tensor operations.

Function Classes:
    - Conv1d: 1D convolution functional
    - Conv2d: 2D convolution functional
    - ConvTranspose2d: 2D transposed convolution functional
    - DepthwiseConv2d: Depthwise 2D convolution functional

Helper Functions:
    - im2col_2d: Image to column transformation
    - col2im_2d: Column to image transformation
    - conv1d, conv2d, conv_transpose2d: Functional interfaces
"""

import numpy as np
from typing import Tuple, Union, Optional

from python.foundations import Function, convert_to_function
from python.foundations.functionals import _unbroadcast

# Global flag for gradient tracking (set by no_grad context)
_no_grad = False

# =============================================================================
# Helper Functions
# =============================================================================

def im2col_2d(
    x: np.ndarray,
    kernel_h: int,
    kernel_w: int,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
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
        padding: (padding_h, padding_w)
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
    padding: Tuple[int, int],
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
        padding: (padding_h, padding_w)
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


# =============================================================================
# Function Classes
# =============================================================================

class Conv1d(Function):
    """
    1D Convolution functional operation for autograd.

    Forward:
        y[n, c_out, l] = Σ_{c_in} Σ_{k} x[n, c_in, l*stride + k] * w[c_out, c_in, k] + b[c_out]
    """
    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> np.ndarray:
        B, C, L = x.shape
        C_out, _, K = weight.shape

        # Pad input
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
        L_padded = x.shape[2]

        L_out = (L_padded - dilation * (K - 1) - 1) // stride + 1

        indices = np.arange(L_out)[:, None] * stride + np.arange(K) * dilation
        patches = x[:, :, indices]  # B, C, L_out, K
        patches = patches.transpose(0, 2, 1, 3).reshape(B, L_out, -1)  # B, L_out, C*K

        output = patches @ weight.reshape(C_out, -1).T  # B, L_out, C_out
        if bias is not None:
            output += bias
        output = output.transpose(0, 2, 1)  # B, C_out, L_out

        global _no_grad
        if not _no_grad:
            self.stride, self.padding, self.dilation = stride, padding, dilation
            self.C, self.K, self.L = C, K, L
            self.L_padded = L_padded
            self.weight = weight  # Store original 3D shape
            self.x_padded = x  # Store padded input (B, C, L_padded)
            self.bias_shape = bias.shape if bias is not None else None

        return output

    def backward(
        self, grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        B, C_out, L_out = grad_output.shape

        # grad_bias
        grad_bias = (
            grad_output.sum(axis=(0, 2)) if self.bias_shape is not None else None
        )

        # grad_weight
        indices = (
            np.arange(L_out)[:, None] * self.stride + np.arange(self.K) * self.dilation
        )
        patches = self.x_padded[:, :, indices]  # B, C, L_out, K
        grad_weight = np.einsum("bol,bilk->oik", grad_output, patches)

        # grad_x: dilate for stride
        if self.stride > 1:
            grad_dilated = np.zeros((B, C_out, (L_out - 1) * self.stride + 1))
            grad_dilated[:, :, :: self.stride] = grad_output
        else:
            grad_dilated = grad_output

        # Pad by effective kernel size - 1
        pad_amount = self.dilation * (self.K - 1)
        grad_padded = np.pad(grad_dilated, ((0, 0), (0, 0), (pad_amount, pad_amount)))

        # Full convolution with flipped weight
        indices = np.arange(self.L_padded)[:, None] + np.arange(self.K) * self.dilation
        grad_patches = grad_padded[:, :, indices]  # B, C_out, L_padded, K
        weight_flipped = self.weight[:, :, ::-1]
        grad_x_padded = np.einsum("bolk,oik->bil", grad_patches, weight_flipped)

        # Remove padding
        if self.padding > 0:
            grad_x = grad_x_padded[:, :, self.padding : -self.padding]
        else:
            grad_x = grad_x_padded

        return grad_x, grad_weight, grad_bias


class Conv2d(Function):
    """
    2D Convolution functional operation for autograd.

    Forward:
        y[n,c_out,h,w] = Σ_{c_in,kh,kw} x[n,c_in,h*s+kh*d,w*s+kw*d] * w[c_out,c_in,kh,kw] + b[c_out]
    """

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1
    ) -> np.ndarray:
        """
        Compute 2D convolution using im2col.

        Args:
            x: Input (batch_size, in_channels, height, width)
            weight: Kernel (out_channels, in_channels/groups, kernel_h, kernel_w)
            bias: Optional bias (out_channels,)
            stride: Convolution stride
            padding: Zero-padding
            dilation: Kernel dilation
            groups: Number of groups for grouped convolution

        Returns:
            Output (batch_size, out_channels, height_out, width_out)
        """
        raise NotImplementedError(
            "TODO: Implement Conv2d forward\n"
            "Hint:\n"
            "  global _no_grad\n"
            "  \n"
            "  # Normalize parameters to tuples\n"
            "  stride = (stride, stride) if isinstance(stride, int) else stride\n"
            "  padding = (padding, padding) if isinstance(padding, int) else padding\n"
            "  dilation = (dilation, dilation) if isinstance(dilation, int) else dilation\n"
            "  \n"
            "  if not _no_grad:\n"
            "      self.x = x\n"
            "      self.weight = weight\n"
            "      self.bias = bias\n"
            "      self.stride = stride\n"
            "      self.padding = padding\n"
            "      self.dilation = dilation\n"
            "      self.groups = groups\n"
            "  \n"
            "  # Pad input\n"
            "  # Use im2col to extract patches\n"
            "  # Reshape weight for matrix multiplication\n"
            "  # output = weight_matrix @ col_matrix + bias\n"
            "  # Reshape to output dimensions"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compute gradients for 2D convolution.

        Args:
            grad_output: Gradient w.r.t. output (batch, out_channels, h_out, w_out)

        Returns:
            Tuple of (grad_x, grad_weight, grad_bias)
        """
        raise NotImplementedError(
            "TODO: Implement Conv2d backward\n"
            "Hint:\n"
            "  # grad_bias = grad_output.sum(axis=(0, 2, 3))\n"
            "  \n"
            "  # grad_weight: correlate input with grad_output\n"
            "  # Use im2col on input, reshape grad_output\n"
            "  # grad_weight = grad_output_reshaped @ col_matrix.T\n"
            "  \n"
            "  # grad_x: convolve grad_output with flipped weights\n"
            "  # Use col2im to convert back to image format"
        )


class ConvTranspose2d(Function):
    """
    2D Transposed Convolution (Deconvolution) functional operation.

    The forward pass is mathematically equivalent to the backward pass
    of Conv2d with respect to the input.
    """

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1
    ) -> np.ndarray:
        """
        Compute transposed 2D convolution.

        Args:
            x: Input (batch_size, in_channels, height, width)
            weight: Kernel (in_channels, out_channels/groups, kernel_h, kernel_w)
            bias: Optional bias (out_channels,)
            stride: Convolution stride
            padding: Zero-padding
            output_padding: Additional padding for output
            dilation: Kernel dilation
            groups: Number of groups

        Returns:
            Upsampled output (batch_size, out_channels, height_out, width_out)
        """
        raise NotImplementedError(
            "TODO: Implement ConvTranspose2d forward\n"
            "Hint: Input dilation approach:\n"
            "  1. Dilate input: insert (stride-1) zeros between elements\n"
            "  2. Pad with kernel_size - 1 - original_padding\n"
            "  3. Perform regular convolution with flipped weights"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compute gradients for transposed 2D convolution.

        Returns:
            Tuple of (grad_x, grad_weight, grad_bias)
        """
        raise NotImplementedError(
            "TODO: Implement ConvTranspose2d backward\n"
            "Hint: The backward of transposed conv is regular conv"
        )


class DepthwiseConv2d(Function):
    """
    Depthwise 2D Convolution functional operation.

    Performs separate convolution for each input channel.
    Equivalent to grouped convolution with groups=in_channels.
    """

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1
    ) -> np.ndarray:
        """
        Compute depthwise 2D convolution.

        Args:
            x: Input (batch_size, channels, height, width)
            weight: Kernel (channels, 1, kernel_h, kernel_w)
            bias: Optional bias (channels,)

        Returns:
            Output (batch_size, channels, height_out, width_out)
        """
        raise NotImplementedError(
            "TODO: Implement DepthwiseConv2d forward\n"
            "Hint: Process each channel independently or use Conv2d with groups=in_channels"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute gradients for depthwise 2D convolution."""
        raise NotImplementedError("TODO: Implement DepthwiseConv2d backward")


class PointwiseConv2d(Function):
    """
    Pointwise (1x1) Convolution functional operation.

    Mixes information across channels without spatial computation.
    """

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute pointwise (1×1) convolution.

        Args:
            x: Input (batch_size, in_channels, height, width)
            weight: Kernel (out_channels, in_channels, 1, 1)
            bias: Optional bias (out_channels,)

        Returns:
            Output (batch_size, out_channels, height, width)
        """
        raise NotImplementedError(
            "TODO: Implement PointwiseConv2d forward\n"
            "Hint: 1×1 conv is just matrix multiply per spatial position:\n"
            "  # Reshape: (B, C_in, H, W) -> (B*H*W, C_in)\n"
            "  # Multiply: output = input @ weight.squeeze().T + bias\n"
            "  # Reshape back: (B*H*W, C_out) -> (B, C_out, H, W)"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute gradients for pointwise convolution."""
        raise NotImplementedError("TODO: Implement PointwiseConv2d backward")


# =============================================================================
# Functional Interfaces (using convert_to_function)
# =============================================================================

conv1d = convert_to_function(Conv1d)
conv2d = convert_to_function(Conv2d)
conv_transpose2d = convert_to_function(ConvTranspose2d)
depthwise_conv2d = convert_to_function(DepthwiseConv2d)
pointwise_conv2d = convert_to_function(PointwiseConv2d)
