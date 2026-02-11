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
        return_indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute 1D max pooling.

        Args:
            x: Input (batch_size, channels, length)
            kernel_size: Pooling window size
            stride: Step size (defaults to kernel_size)
            padding: Zero-padding
            dilation: Dilation factor
            ceil_mode: Use ceiling for output size
            return_indices: Return indices of max values

        Returns:
            Pooled output, and optionally max indices
        """
        raise NotImplementedError(
            "TODO: Implement MaxPool1d forward\n"
            "Hint:\n"
            "  global _no_grad\n"
            "  stride = stride if stride is not None else kernel_size\n"
            "  \n"
            "  if not _no_grad:\n"
            "      self.x = x\n"
            "      self.kernel_size = kernel_size\n"
            "      self.stride = stride\n"
            "      self.padding = padding\n"
            "  \n"
            "  # Pad input if needed\n"
            "  # Extract windows using sliding window view\n"
            "  # Take max over kernel dimension\n"
            "  # Store indices for backward pass"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient for 1D max pooling.

        Gradient flows only through the max elements.
        """
        raise NotImplementedError(
            "TODO: Implement MaxPool1d backward\n"
            "Hint: Use stored indices to route gradients to max positions"
        )


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
        raise NotImplementedError(
            "TODO: Implement MaxPool2d forward\n"
            "Hint:\n"
            "  global _no_grad\n"
            "  \n"
            "  # Normalize to tuples\n"
            "  kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size\n"
            "  stride = stride if stride is not None else kernel_size\n"
            "  stride = (stride, stride) if isinstance(stride, int) else stride\n"
            "  padding = (padding, padding) if isinstance(padding, int) else padding\n"
            "  \n"
            "  if not _no_grad:\n"
            "      self.x = x\n"
            "      self.kernel_size = kernel_size\n"
            "      self.stride = stride\n"
            "      self.padding = padding\n"
            "  \n"
            "  # Pad input\n"
            "  # Extract patches using as_strided or loop\n"
            "  # Take max over kernel dimensions\n"
            "  # Store argmax indices for backward"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient for 2D max pooling.

        Gradient flows only through max elements (sparse gradient).
        """
        raise NotImplementedError(
            "TODO: Implement MaxPool2d backward\n"
            "Hint:\n"
            "  # Create zero gradient for input\n"
            "  # Use stored indices to place grad_output values at max positions\n"
            "  # Handle overlapping windows by accumulating gradients"
        )


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
        raise NotImplementedError(
            "TODO: Implement AvgPool1d forward\n"
            "Hint:\n"
            "  # Similar to MaxPool1d but take mean instead of max\n"
            "  # If count_include_pad=False, divide by actual (non-padded) count"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient for 1D average pooling.

        Gradient is distributed evenly to all elements in each window.
        """
        raise NotImplementedError(
            "TODO: Implement AvgPool1d backward\n"
            "Hint: Gradient is 1/kernel_size for each element in the window"
        )


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
        """
        Compute 2D average pooling.

        Args:
            x: Input (batch_size, channels, height, width)
            kernel_size: Pooling window size
            stride: Step size (defaults to kernel_size)
            padding: Zero-padding
            ceil_mode: Use ceiling for output size
            count_include_pad: Include padding in average

        Returns:
            Pooled output (batch, channels, h_out, w_out)
        """
        raise NotImplementedError(
            "TODO: Implement AvgPool2d forward\n"
            "Hint:\n"
            "  global _no_grad\n"
            "  \n"
            "  # Normalize to tuples\n"
            "  # Pad input\n"
            "  # Extract patches\n"
            "  # Take mean over kernel dimensions"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient for 2D average pooling.

        Gradient is distributed evenly to all elements in each window.
        """
        raise NotImplementedError(
            "TODO: Implement AvgPool2d backward\n"
            "Hint:\n"
            "  # Each output gradient is divided by kernel_h * kernel_w\n"
            "  # and distributed to all input positions in the window"
        )


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
        return_indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute adaptive max pooling.

        Args:
            x: Input (batch_size, channels, height, width)
            output_size: Target output size (h, w) or single int for square
            return_indices: Return indices of max values

        Returns:
            Pooled output with target size, and optionally indices
        """
        raise NotImplementedError(
            "TODO: Implement AdaptiveMaxPool2d forward\n"
            "Hint:\n"
            "  # Convert output_size to tuple\n"
            "  output_size = (output_size, output_size) if isinstance(output_size, int) else output_size\n"
            "  \n"
            "  # Compute kernel_size and stride to achieve output_size\n"
            "  # kernel_h = ceil(H / out_h), stride_h = floor(H / out_h)\n"
            "  # Apply max pooling with computed parameters"
        )

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

class GlobalAvgPool2d(Function):
    """
    Global Average Pooling 2D functional operation.

    Reduces spatial dimensions to 1x1 by taking mean.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute global average pooling.

        Args:
            x: Input (batch_size, channels, height, width)

        Returns:
            Output (batch_size, channels) or (batch_size, channels, 1, 1)
        """
        raise NotImplementedError(
            "TODO: Implement GlobalAvgPool2d forward\n"
            "Hint:\n"
            "  global _no_grad\n"
            "  if not _no_grad:\n"
            "      self.x_shape = x.shape\n"
            "  return x.mean(axis=(2, 3))"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient for global average pooling.

        Each output gradient is divided equally among all spatial positions.
        """
        raise NotImplementedError(
            "TODO: Implement GlobalAvgPool2d backward\n"
            "Hint:\n"
            "  batch, channels, h, w = self.x_shape\n"
            "  # Distribute gradient equally: grad / (h * w)\n"
            "  grad_x = grad_output[:, :, np.newaxis, np.newaxis] / (h * w)\n"
            "  return (np.broadcast_to(grad_x, self.x_shape).copy(),)"
        )


class GlobalMaxPool2d(Function):
    """
    Global Max Pooling 2D functional operation.

    Reduces spatial dimensions to 1x1 by taking max.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute global max pooling.

        Args:
            x: Input (batch_size, channels, height, width)

        Returns:
            Output (batch_size, channels)
        """
        raise NotImplementedError(
            "TODO: Implement GlobalMaxPool2d forward\n"
            "Hint:\n"
            "  global _no_grad\n"
            "  if not _no_grad:\n"
            "      self.x = x\n"
            "      # Store argmax indices for backward\n"
            "      self.max_indices = x.reshape(x.shape[0], x.shape[1], -1).argmax(axis=2)\n"
            "  return x.max(axis=(2, 3))"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Compute gradient for global max pooling."""
        raise NotImplementedError(
            "TODO: Implement GlobalMaxPool2d backward\n"
            "Hint: Route gradient only to max positions using stored indices"
        )


# =============================================================================
# Functional Interfaces
# =============================================================================

max_pool1d = convert_to_function(MaxPool1d)
max_pool2d = convert_to_function(MaxPool2d)
avg_pool1d = convert_to_function(AvgPool1d)
avg_pool2d = convert_to_function(AvgPool2d)
adaptive_max_pool2d = convert_to_function(AdaptiveMaxPool2d)
adaptive_avg_pool2d = convert_to_function(AdaptiveAvgPool2d)
global_avg_pool2d = convert_to_function(GlobalAvgPool2d)
global_max_pool2d = convert_to_function(GlobalMaxPool2d)
