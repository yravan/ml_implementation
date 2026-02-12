"""
Pooling Functional Operations
=============================

This module provides functional operations for pooling layers.
Function classes handle the forward/backward computation with np.ndarray,
while Module classes in pooling.py wrap these for Tensor operations.

Function Classes:
    - MaxPool1d: 1D max pooling functional
    - MaxPool2d: 2D max pooling functional
    - AvgPool1d: 1D average pooling functional
    - AvgPool2d: 2D average pooling functional
    - AdaptiveMaxPool2d: Adaptive max pooling functional
    - AdaptiveAvgPool2d: Adaptive average pooling functional

Helper Functions:
    - max_pool1d, max_pool2d: Max pooling interfaces
    - avg_pool1d, avg_pool2d: Average pooling interfaces
    - adaptive_max_pool2d, adaptive_avg_pool2d: Adaptive pooling interfaces
    - global_avg_pool2d, global_max_pool2d: Global pooling interfaces
"""

import numpy as np
from typing import Tuple, Union, Optional

from python.foundations import Function, convert_to_function

# Global flag for gradient tracking
_no_grad = False


# =============================================================================
# Helper Functions
# =============================================================================

def compute_pooling_output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
) -> int:
    """
    Compute output size for pooling.

    Formula:
        output = floor((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
    """
    numerator = input_size + 2*padding - dilation*(kernel_size - 1) - 1
    if ceil_mode:
        output = int(np.ceil(numerator / stride)) + 1
    else:
        output = numerator // stride + 1
    return output


# =============================================================================
# Max Pooling Function Classes
# =============================================================================

class MaxPool1d(Function):
    """
    Max Pooling 1D functional operation.

    Applies max pooling over a 1D input signal.
    """

    def forward(
        self,
        x: np.ndarray,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        dilation: int = 1,
        ceil_mode: bool = False,
    ) -> np.ndarray:
        """
        Compute 1D max pooling.

        Args:
            x: Input (batch_size, channels, length)
            kernel_size: Pooling window size
            stride: Step size (defaults to kernel_size)
            padding: Zero-padding
            dilation: Dilation factor
            ceil_mode: Use ceiling for output size

        Returns:
            Pooled output
        """
        assert x.ndim == 3
        B, C, L = x.shape
        input_size = x.shape[-1]
        output_size = compute_pooling_output_size(input_size, kernel_size, stride, padding, dilation, ceil_mode)

        if padding > 0:
            padded_x = np.zeros((B, C, L + 2 * padding))
            padded_x[:, :, padding:-padding] = x
            x = padded_x

        patch_indices = np.arange(output_size)[:, None] * stride + np.arange(kernel_size)[None, :] * dilation # L_out x K
        patches = x[:, :, patch_indices] # B, C, L_out, K
        output = patches.max(axis=-1) # B, C, L_out
        output_indices = patches.argmax(axis=-1) * dilation + np.arange(output_size) * stride

        global _no_grad
        if not _no_grad:
            # store which indices the output came from
            self.shape = x.shape
            self.output_indices = output_indices # B, C, L_out
            self.padding = padding

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient for 1D max pooling.

        Gradient flows only through the max elements.
        """
        B, C, L_out = grad_output.shape
        grad_x = np.zeros(self.shape)
        b_idx = np.arange(B)[:, None, None]
        c_idx = np.arange(C)[None, :, None]
        np.add.at(grad_x, (b_idx, c_idx, self.output_indices), grad_output)
        if self.padding > 0:
            grad_x = grad_x[:, :, self.padding:-self.padding]
        return (grad_x,)


class MaxPool2d(Function):
    """
    Max Pooling 2D functional operation.

    Applies max pooling over a 2D input (images/feature maps).
    """

    def forward(
        self,
        x: np.ndarray,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute 2D max pooling.

        Args:
            x: Input (batch_size, channels, height, width)
            kernel_size: Pooling window size
            stride: Step size (defaults to kernel_size)
            padding: Zero-padding
            dilation: Dilation factor
            ceil_mode: Use ceiling for output size
            return_indices: Return indices of max values

        Returns:
            Pooled output (batch, channels, h_out, w_out), and optionally indices
        """
        assert x.ndim == 4
        B, C, H, W = x.shape
        H_out = compute_pooling_output_size(H, kernel_size, stride, padding, dilation, ceil_mode)
        W_out = compute_pooling_output_size(W, kernel_size, stride, padding, dilation, ceil_mode)

        if padding > 0:
            padded_x = np.zeros((B, C, H + 2 * padding, W + 2 * padding))
            padded_x[:, :, padding:-padding, padding:-padding] = x
            x = padded_x

        patch_h_indices = np.arange(H_out)[:, None] * stride + np.arange(kernel_size)[None, :] * dilation # H_out x K
        patch_w_indices = np.arange(W_out)[:, None] * stride + np.arange(kernel_size)[None, :] * dilation # W_out x K
        patches = x[:, :, patch_h_indices[:, :, None, None], patch_w_indices[None, None, :, :]] # B, C, H_out, K, W_out, K
        patches = patches.transpose((0, 1, 2, 4, 3, 5)) # B, C, H_out, W_out, K, K
        flat_patches = patches.reshape(B, C, H_out, W_out, -1)  # B, C, H_out, W_out, K*K
        flat_idx = flat_patches.argmax(axis=-1)                   # B, C, H_out, W_out
        output = flat_patches.max(axis=-1)

        # Convert flat index to 2D kernel position
        kh = flat_idx // kernel_size
        kw = flat_idx % kernel_size

        # Convert to input coordinates
        h_indices = np.arange(H_out)[None, None, :, None] * stride + kh * dilation
        w_indices = np.arange(W_out)[None, None, None, :] * stride + kw * dilation
        # B, C, H_out, W_out, 2
        global _no_grad
        if not _no_grad:
            # store which indices the output came from
            self.shape = x.shape
            self.h_indices = h_indices # 1, 1, H_out, W_out
            self.w_indices = w_indices # 1, 1, H_out, W_out
            self.padding = padding

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient for 2D max pooling.

        Gradient flows only through max elements (sparse gradient).
        """
        grad_x = np.zeros(self.shape)
        b_idx = np.arange(grad_output.shape[0])[:, None, None, None]
        c_idx = np.arange(grad_output.shape[1])[None, :, None, None]
        np.add.at(grad_x, (b_idx, c_idx, self.h_indices, self.w_indices), grad_output)
        if self.padding > 0:
            grad_x = grad_x[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return (grad_x,)


# =============================================================================
# Average Pooling Function Classes
# =============================================================================

class AvgPool1d(Function):
    """
    Average Pooling 1D functional operation.

    Applies average pooling over a 1D input signal.
    """

    def forward(
        self,
        x: np.ndarray,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True
    ) -> np.ndarray:
        """
        Compute 1D average pooling.

        Args:
            x: Input (batch_size, channels, length)
            kernel_size: Pooling window size
            stride: Step size (defaults to kernel_size)
            padding: Zero-padding
            ceil_mode: Use ceiling for output size
            count_include_pad: Include padding in average computation

        Returns:
            Pooled output (batch, channels, length_out)
        """
        assert x.ndim == 3
        B, C, L = x.shape
        input_size = x.shape[-1]
        output_size = compute_pooling_output_size(input_size, kernel_size, stride, padding, 1, ceil_mode)

        if padding > 0:
            padded_x = np.zeros((B, C, L + 2 * padding))
            padded_x[:, :, padding:-padding] = x
            x = padded_x

        patch_indices = np.arange(output_size)[:, None] * stride + np.arange(kernel_size)[None, :] # L_out x K
        patches = x[:, :, patch_indices] # B, C, L_out, K
        if count_include_pad:
            output = patches.mean(axis=-1) # B, C, L_out
        else:
            mask = np.ones((B, C, L))
            if padding > 0:
                mask = np.pad(mask, ((0,0), (0,0), (padding, padding)))
            mask_patches = mask[:, :, patch_indices]  # B, C, L_out, K
            counts = mask_patches.sum(axis=-1)        # B, C, L_out
            output = patches.sum(axis=-1) / counts

        global _no_grad
        if not _no_grad:
            self.kernel_size = kernel_size
            self.padding = padding
            self.shape = x.shape
            self.stride = stride
            if count_include_pad:
                self.divisor = kernel_size
            else:
                self.divisor = counts  # (B, C, L_out)

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient for 1D average pooling.

        Gradient is distributed evenly to all elements in each window.
        """
        B, C, L_out = grad_output.shape
        grad_x = np.zeros(self.shape)
        patch_indices = np.arange(L_out)[:, None] * self.stride + np.arange(self.kernel_size)[None, :]  # L_out, K
        scaled_grad = (grad_output / self.divisor)[:, :, :, None]  # B, C, L_out, 1 (broadcast over K)
        np.add.at(grad_x, (slice(None), slice(None), patch_indices), scaled_grad)
        if self.padding > 0:
            grad_x = grad_x[:, :, self.padding:-self.padding]
        return (grad_x,)


class AvgPool2d(Function):
    """
    Average Pooling 2D functional operation.

    Applies average pooling over a 2D input (images/feature maps).
    """

    def forward(
        self,
        x: np.ndarray,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True
    ) -> np.ndarray:
        assert x.ndim == 4
        B, C, H, W = x.shape

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        H_out = compute_pooling_output_size(H, kernel_size[0], stride[0], padding[0], 1, ceil_mode)
        W_out = compute_pooling_output_size(W, kernel_size[1], stride[1], padding[1], 1, ceil_mode)

        if padding[0] > 0 or padding[1] > 0:
            x = np.pad(x, ((0,0), (0,0), (padding[0], padding[0]), (padding[1], padding[1])))

        h_idx = np.arange(H_out)[:, None] * stride[0] + np.arange(kernel_size[0])[None, :]  # H_out, Kh
        w_idx = np.arange(W_out)[:, None] * stride[1] + np.arange(kernel_size[1])[None, :]  # W_out, Kw

        patches = x[:, :, h_idx[:, :, None, None], w_idx[None, None, :, :]]  # B, C, H_out, Kh, W_out, Kw
        patches = patches.transpose(0, 1, 2, 4, 3, 5)  # B, C, H_out, W_out, Kh, Kw

        if count_include_pad:
            output = patches.mean(axis=(-2, -1))
            divisor = kernel_size[0] * kernel_size[1]
        else:
            mask = np.ones((B, C, H, W))
            if padding[0] > 0 or padding[1] > 0:
                mask = np.pad(mask, ((0,0), (0,0), (padding[0], padding[0]), (padding[1], padding[1])))
            mask_patches = mask[:, :, h_idx[:, :, None, None], w_idx[None, None, :, :]]
            mask_patches = mask_patches.transpose(0, 1, 2, 4, 3, 5)
            counts = mask_patches.sum(axis=(-2, -1))  # B, C, H_out, W_out
            output = patches.sum(axis=(-2, -1)) / counts
            divisor = counts

        global _no_grad
        if not _no_grad:
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.shape = x.shape  # padded shape
            self.orig_H = H
            self.orig_W = W
            self.divisor = divisor

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        B, C, H_out, W_out = grad_output.shape
        grad_x = np.zeros(self.shape)

        h_idx = np.arange(H_out)[:, None] * self.stride[0] + np.arange(self.kernel_size[0])[None, :]  # H_out, Kh
        w_idx = np.arange(W_out)[:, None] * self.stride[1] + np.arange(self.kernel_size[1])[None, :]  # W_out, Kw

        scaled_grad = (grad_output / self.divisor)[:, :, :, :, None, None]  # B, C, H_out, W_out, 1, 1
        # Target indices: (H_out, Kh, W_out, Kw) via broadcasting
        h_full = h_idx[:, :, None, None]  # H_out, Kh, 1, 1
        w_full = w_idx[None, None, :, :]  # 1, 1, W_out, Kw
        # scaled_grad needs shape B, C, H_out, Kh, W_out, Kw
        scaled_grad = scaled_grad.transpose(0, 1, 2, 4, 3, 5)  # B, C, H_out, 1, W_out, 1 → broadcast

        np.add.at(grad_x, (slice(None), slice(None), h_full, w_full), scaled_grad)

        if self.padding[0] > 0 or self.padding[1] > 0:
            grad_x = grad_x[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]

        return (grad_x,)

# =============================================================================
# Adaptive Pooling Function Classes
# =============================================================================

class AdaptiveMaxPool2d(Function):
    """
    Adaptive Max Pooling 2D functional operation.

    Automatically computes kernel size and stride to achieve target output size.
    """

    def forward(
        self,
        x: np.ndarray,
        output_size: Union[int, Tuple[int, int]],
    ) -> np.ndarray:
        """
        Compute adaptive max pooling.

        Args:
            x: Input (batch_size, channels, height, width)
            output_size: Target output size (h, w) or single int for square
            return_indices: Return indices of max values

        Returns:
            Pooled output with target size, and optionally indices
        """
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        output_h, output_w = output_size

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Compute gradient for adaptive max pooling."""
        raise NotImplementedError("TODO: Implement AdaptiveMaxPool2d backward")


class AdaptiveAvgPool2d(Function):
    """
    Adaptive Average Pooling 2D functional operation.

    Automatically computes kernel size and stride to achieve target output size.
    """

    def forward(
        self,
        x: np.ndarray,
        output_size: Union[int, Tuple[int, int]]
    ) -> np.ndarray:
        """
        Compute adaptive average pooling.

        Args:
            x: Input (batch_size, channels, height, width)
            output_size: Target output size (h, w) or single int for square

        Returns:
            Pooled output with target size
        """
        raise NotImplementedError(
            "TODO: Implement AdaptiveAvgPool2d forward\n"
            "Hint:\n"
            "  # For output_size=1, this is global average pooling:\n"
            "  # return x.mean(axis=(2, 3), keepdims=True)\n"
            "  \n"
            "  # For other sizes, compute kernel and stride adaptively"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Compute gradient for adaptive average pooling."""
        raise NotImplementedError("TODO: Implement AdaptiveAvgPool2d backward")


# =============================================================================
# Global Pooling Function Classes
# =============================================================================

class GlobalAvgPool1d(Function):
    """
    Global Average Pooling !D functional operation.

    Reduces spatial dimensions to 1x1 by taking mean.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 3
        global _no_grad
        if not _no_grad:
            self.shape = x.shape
        return x.mean(axis=-1)  # B, C

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        L = self.shape[2]
        grad_x = grad_output[:, :, None] / (L)
        return (np.broadcast_to(grad_x, self.shape).copy(),)


class GlobalMaxPool1d(Function):
    """
    Global Max Pooling 2D functional operation.

    Reduces spatial dimensions to 1x1 by taking max.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 3
        B, C, L = x.shape
        output = x.max(axis=2)  # B, C

        global _no_grad
        if not _no_grad:
            self.shape = x.shape
            self.indices = x.argmax(axis=2)

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        grad_x = np.zeros(self.shape)
        b_idx = np.arange(grad_output.shape[0])[:, None]
        c_idx = np.arange(grad_output.shape[1])[None, :]
        grad_x[b_idx, c_idx, self.indices] = grad_output
        return (grad_x,)

class GlobalAvgPool2d(Function):
    """
    Global Average Pooling 2D functional operation.

    Reduces spatial dimensions to 1x1 by taking mean.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 4
        global _no_grad
        if not _no_grad:
            self.shape = x.shape
        return x.mean(axis=(2, 3))  # B, C

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        H, W = self.shape[2], self.shape[3]
        grad_x = grad_output[:, :, None, None] / (H * W)
        return (np.broadcast_to(grad_x, self.shape).copy(),)


class GlobalMaxPool2d(Function):
    """
    Global Max Pooling 2D functional operation.

    Reduces spatial dimensions to 1x1 by taking max.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 4
        B, C, H, W = x.shape
        output = x.max(axis=(2, 3))  # B, C

        global _no_grad
        if not _no_grad:
            self.shape = x.shape
            flat = x.reshape(B, C, -1)
            flat_idx = flat.argmax(axis=2)  # B, C
            self.h_indices = flat_idx // W
            self.w_indices = flat_idx % W

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        grad_x = np.zeros(self.shape)
        b_idx = np.arange(grad_output.shape[0])[:, None]
        c_idx = np.arange(grad_output.shape[1])[None, :]
        grad_x[b_idx, c_idx, self.h_indices, self.w_indices] = grad_output
        return (grad_x,)


# =============================================================================
# Functional Interfaces
# =============================================================================

max_pool1d = convert_to_function(MaxPool1d)
max_pool2d = convert_to_function(MaxPool2d)
avg_pool1d = convert_to_function(AvgPool1d)
avg_pool2d = convert_to_function(AvgPool2d)
adaptive_max_pool2d = convert_to_function(AdaptiveMaxPool2d)
adaptive_avg_pool2d = convert_to_function(AdaptiveAvgPool2d)
global_avg_pool1d = convert_to_function(GlobalAvgPool1d)
global_max_pool1d = convert_to_function(GlobalMaxPool1d)
global_avg_pool2d = convert_to_function(GlobalAvgPool2d)
global_max_pool2d = convert_to_function(GlobalMaxPool2d)
