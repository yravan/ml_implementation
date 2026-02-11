"""
Pooling Layers for Neural Networks
===================================

This module consolidates all pooling operations including:
- Max Pooling (1D, 2D, Adaptive)
- Average Pooling (2D, Adaptive)
- Global Pooling (Average, Max)
- Spatial Pyramid Pooling
- Mixed and Stochastic variants

All modules inherit from Module and process Tensor inputs.
Backward passes are handled by Function classes in functional.py via autograd.
"""

from typing import Optional, Union, Tuple, List
from python.foundations import Tensor
from .module import Module, Parameter


# ============================================================================
# Max Pooling Classes
# ============================================================================

class MaxPool1d(Module):
    """
    Max Pooling 1D layer.

    Applies max pooling operation to 1D input (sequences, time series).
    Divides input into pooling windows and outputs maximum value from each window.

    Attributes:
        kernel_size: Pooling window size
        stride: Step size between pooling windows
        padding: Zero-padding to apply
        dilation: Spacing between kernel elements
        ceil_mode: If True, use ceiling instead of floor for size calculation
        return_indices: If True, return indices of max values
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Optional[Union[int, Tuple[int]]] = None,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ):
        """
        Initialize max pooling 1D layer.

        Args:
            kernel_size: Size of pooling window
            stride: Step size between windows. If None, defaults to kernel_size
            padding: Zero-padding to apply before pooling
            dilation: Spacing between kernel elements
            ceil_mode: If True, use ceiling for output size calculation
            return_indices: If True, forward() returns (output, indices)

        Raises:
            ValueError: If kernel_size <= 0
            ValueError: If stride <= 0
            ValueError: If padding < 0
            ValueError: If dilation <= 0
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply max pooling to input.

        Args:
            x: Input tensor of shape (batch_size, channels, length)

        Returns:
            Pooled tensor of shape (batch_size, channels, length_out)
            If return_indices=True, returns (output, max_indices)

        Raises:
            ValueError: If x is not 3D (batch, channels, length)
        """
        raise NotImplementedError(
            "TODO: Implement forward pass\n"
            "1. Validate input is 3D: (batch, channels, length)\n"
            "2. Use torch.nn.functional.max_pool1d with appropriate parameters:\n"
            "   F.max_pool1d(x, kernel_size, stride, padding, dilation,\n"
            "                ceil_mode, return_indices)\n"
            "3. If return_indices=False: return output\n"
            "4. If return_indices=True: return (output, indices)\n"
            "\n"
            "HINT: torch.nn.functional has max_pool1d ready to use"
        )

    def compute_output_size(
        self, input_size: Union[int, Tuple[int]]
    ) -> Union[int, Tuple[int]]:
        """
        Compute output spatial size given input size.

        Args:
            input_size: Input spatial size(s)

        Returns:
            Output spatial size(s)

        Formula:
            output = floor((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
            OR if ceil_mode: use ceil instead of floor
        """
        raise NotImplementedError(
            "TODO: Compute output size\n"
            "1. If input_size is int, handle single dimension\n"
            "2. Use formula:\n"
            "   output = (input + 2*padding - dilation*(kernel-1) - 1) / stride + 1\n"
            "3. Apply floor or ceil based on self.ceil_mode\n"
            "4. Return output size"
        )

    def extra_repr(self) -> str:
        """Return string representation with parameters."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, ceil_mode={self.ceil_mode}"


class MaxPool2d(Module):
    """
    Max Pooling 2D layer.

    Applies max pooling operation to spatial input (typically images).
    Divides input into pooling windows and outputs maximum value from each window.

    Example:
        >>> pool = MaxPool2d(kernel_size=2, stride=2)
        >>> x = torch.randn(batch_size, 3, 224, 224)
        >>> output = pool(x)
        >>> assert output.shape == (batch_size, 3, 112, 112)

    Attributes:
        kernel_size: Pooling window size
        stride: Step size between pooling windows
        padding: Zero-padding to apply
        dilation: Spacing between kernel elements (for dilated pooling)
        ceil_mode: If True, use ceiling instead of floor for size calculation
        return_indices: If True, return indices of max values (useful for unpooling)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, ...]] = 2,
        stride: Optional[Union[int, Tuple[int, ...]]] = None,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ):
        """
        Initialize max pooling layer.

        Args:
            kernel_size: Size of pooling window (single int or tuple for h,w)
            stride: Step size between windows. If None, defaults to kernel_size
            padding: Zero-padding to apply before pooling
            dilation: Spacing between kernel elements (advanced)
            ceil_mode: If True, use ceiling for output size calculation
            return_indices: If True, forward() returns (output, indices)

        Raises:
            ValueError: If kernel_size <= 0
            ValueError: If stride <= 0
            ValueError: If padding < 0
            ValueError: If dilation <= 0
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply max pooling to input.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Pooled tensor of shape (batch_size, channels, height_out, width_out)
            If return_indices=True, returns (output, max_indices)

        Raises:
            ValueError: If x is not 4D (batch, channels, height, width)
            ValueError: If height or width < kernel_size + padding considerations

        Example:
            >>> pool = MaxPool2d(kernel_size=3, stride=1, padding=1)
            >>> x = torch.randn(2, 64, 28, 28)
            >>> output = pool(x)
            >>> assert output.shape == (2, 64, 28, 28)  # Same spatial size with padding
        """
        raise NotImplementedError(
            "TODO: Implement forward pass\n"
            "1. Validate input is 4D: (batch, channels, height, width)\n"
            "2. Use torch.nn.functional.max_pool2d with appropriate parameters:\n"
            "   F.max_pool2d(x, kernel_size, stride, padding, dilation,\n"
            "                ceil_mode, return_indices)\n"
            "3. If return_indices=False: return output\n"
            "4. If return_indices=True: return (output, indices)\n"
            "\n"
            "HINT: torch.nn.functional has max_pool2d ready to use"
        )

    def compute_output_size(
        self, input_size: Union[int, Tuple[int, int]]
    ) -> Union[int, Tuple[int, int]]:
        """
        Compute output spatial size given input size.

        Args:
            input_size: Input spatial size(s)

        Returns:
            Output spatial size(s)

        Formula:
            output = floor((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
            OR if ceil_mode: use ceil instead of floor

        Example:
            >>> pool = MaxPool2d(kernel_size=3, stride=1, padding=1)
            >>> out_size = pool.compute_output_size(input_size=28)
            >>> assert out_size == 28  # Same size due to padding
        """
        raise NotImplementedError(
            "TODO: Compute output size\n"
            "1. If input_size is int, handle single dimension\n"
            "2. If input_size is tuple, apply formula to each dimension\n"
            "3. Use formula:\n"
            "   output = (input + 2*padding - dilation*(kernel-1) - 1) / stride + 1\n"
            "4. Apply floor or ceil based on self.ceil_mode\n"
            "5. Return output size(s) as same type as input"
        )

    def extra_repr(self) -> str:
        """Return string representation with parameters."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, ceil_mode={self.ceil_mode}"


class MaxPool3d(Module):
    """
    Max Pooling 3D layer.

    Applies max pooling operation to 3D input (volumetric data, video).
    Divides input into pooling windows and outputs maximum value from each window.

    Attributes:
        kernel_size: Pooling window size
        stride: Step size between pooling windows
        padding: Zero-padding to apply
        dilation: Spacing between kernel elements
        ceil_mode: If True, use ceiling instead of floor for size calculation
        return_indices: If True, return indices of max values
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ):
        """
        Initialize max pooling 3D layer.

        Args:
            kernel_size: Size of pooling window
            stride: Step size between windows. If None, defaults to kernel_size
            padding: Zero-padding to apply before pooling
            dilation: Spacing between kernel elements
            ceil_mode: If True, use ceiling for output size calculation
            return_indices: If True, forward() returns (output, indices)
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply max pooling to input.

        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)

        Returns:
            Pooled tensor of shape (batch_size, channels, D_out, H_out, W_out)
            If return_indices=True, returns (output, max_indices)
        """
        raise NotImplementedError(
            "TODO: Implement forward pass\n"
            "1. Validate input is 5D: (batch, channels, depth, height, width)\n"
            "2. Use max_pool3d functional with appropriate parameters\n"
            "3. Return output or (output, indices)"
        )

    def extra_repr(self) -> str:
        """Return string representation with parameters."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AdaptiveMaxPool1d(Module):
    """
    Adaptive Max Pooling 1D.

    Automatically computes kernel and stride to achieve desired output size.

    Attributes:
        output_size: Target output size
        return_indices: If True, return indices of max values
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[int]],
        return_indices: bool = False,
    ):
        """
        Initialize adaptive max pooling 1D.

        Args:
            output_size: Desired output size
            return_indices: If True, forward() returns (output, indices)
        """
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply adaptive max pooling.

        Args:
            x: Input tensor of shape (batch, channels, length)

        Returns:
            Output tensor of shape (batch, channels, output_size)
        """
        raise NotImplementedError(
            "TODO: Implement adaptive max pooling 1D forward pass\n"
            "1. Validate input is 3D\n"
            "2. Use adaptive_max_pool1d functional\n"
            "3. Return output or (output, indices)"
        )


class AdaptiveMaxPool2d(Module):
    """
    Adaptive Max Pooling 2D.

    Unlike MaxPool2d which specifies kernel size, AdaptiveMaxPool2d specifies
    the desired output spatial size. The kernel and stride are automatically
    computed to achieve this output size.

    Useful for:
    - Handling variable input sizes while producing fixed output sizes
    - Compressing spatial dimensions to fixed sizes for downstream layers
    - Global pooling (output_size=1)

    Example:
        >>> pool = AdaptiveMaxPool2d(output_size=(7, 7))
        >>> x = torch.randn(batch_size, channels, height, width)  # Any height/width
        >>> output = pool(x)
        >>> assert output.shape == (batch_size, channels, 7, 7)

        >>> # Global pooling
        >>> global_pool = AdaptiveMaxPool2d(output_size=1)
        >>> output = global_pool(x)
        >>> assert output.shape == (batch_size, channels, 1, 1)

    Attributes:
        output_size: Target output spatial size
        return_indices: If True, return indices of max values
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[int, ...]],
        return_indices: bool = False,
    ):
        """
        Initialize adaptive max pooling.

        Args:
            output_size: Desired output spatial size. Can be:
                - Single int: output will be (output_size, output_size)
                - Tuple of ints: output will be (h, w)
            return_indices: If True, forward() returns (output, indices)

        Raises:
            ValueError: If output_size <= 0
        """
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply adaptive max pooling.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, channels, output_h, output_w)
            If return_indices=True: (output, max_indices)

        Note:
            PyTorch provides torch.nn.functional.adaptive_max_pool2d
            which handles computation efficiently.

        Example:
            >>> pool = AdaptiveMaxPool2d(output_size=7)
            >>> x = torch.randn(32, 256, 32, 32)
            >>> output = pool(x)
            >>> assert output.shape == (32, 256, 7, 7)
        """
        raise NotImplementedError(
            "TODO: Implement adaptive max pooling forward pass\n"
            "1. Validate input is 4D\n"
            "2. Use torch.nn.functional.adaptive_max_pool2d\n"
            "3. If return_indices=True: use return_indices parameter\n"
            "4. Return output or (output, indices)"
        )


class AdaptiveMaxPool3d(Module):
    """
    Adaptive Max Pooling 3D.

    Automatically computes kernel and stride to achieve desired output size
    for volumetric data.

    Attributes:
        output_size: Target output spatial size
        return_indices: If True, return indices of max values
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int, int]],
        return_indices: bool = False,
    ):
        """
        Initialize adaptive max pooling 3D.

        Args:
            output_size: Desired output spatial size (D, H, W)
            return_indices: If True, forward() returns (output, indices)
        """
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply adaptive max pooling.

        Args:
            x: Input tensor of shape (batch, channels, depth, height, width)

        Returns:
            Output tensor of shape (batch, channels, D_out, H_out, W_out)
        """
        raise NotImplementedError(
            "TODO: Implement adaptive max pooling 3D forward pass\n"
            "1. Validate input is 5D\n"
            "2. Use adaptive_max_pool3d functional\n"
            "3. Return output or (output, indices)"
        )


# ============================================================================
# Average Pooling Classes
# ============================================================================

class AvgPool1d(Module):
    """
    Average Pooling 1D layer.

    Applies average pooling operation to 1D input (sequences, time series).
    Divides input into pooling windows and outputs the average (mean) value from each window.

    Useful for smoothing feature maps and reducing spatial dimensions while
    preserving overall statistical information.

    Attributes:
        kernel_size: Pooling window size
        stride: Step size between pooling windows
        padding: Zero-padding to apply
        ceil_mode: If True, use ceiling instead of floor for size calculation
        count_include_pad: If True, include padded positions in average calculation
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Optional[Union[int, Tuple[int]]] = None,
        padding: Union[int, Tuple[int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ):
        """
        Initialize average pooling 1D layer.

        Args:
            kernel_size: Size of pooling window
            stride: Step size between windows. If None, defaults to kernel_size
            padding: Zero-padding to apply before pooling
            ceil_mode: If True, use ceiling for output size calculation
            count_include_pad: If True, padded zeros are included in average

        Raises:
            ValueError: If kernel_size <= 0
            ValueError: If stride <= 0 (when not None)
            ValueError: If padding < 0
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply average pooling to input.

        Args:
            x: Input tensor of shape (batch_size, channels, length)

        Returns:
            Pooled tensor of shape (batch_size, channels, length_out)

        Raises:
            ValueError: If x is not 3D (batch, channels, length)
        """
        raise NotImplementedError(
            "TODO: Implement forward pass\n"
            "1. Validate input is 3D: (batch, channels, length)\n"
            "2. Use torch.nn.functional.avg_pool1d with parameters:\n"
            "   F.avg_pool1d(x, kernel_size, stride, padding,\n"
            "                ceil_mode, count_include_pad)\n"
            "3. Return output"
        )

    def compute_output_size(
        self, input_size: Union[int, Tuple[int]]
    ) -> Union[int, Tuple[int]]:
        """
        Compute output spatial size given input size.

        Args:
            input_size: Input spatial size(s)

        Returns:
            Output spatial size(s)

        Formula:
            output = floor((input + 2*padding - kernel_size) / stride) + 1
            OR if ceil_mode: use ceil instead of floor
        """
        raise NotImplementedError(
            "TODO: Compute output size\n"
            "Apply pooling size formula"
        )

    def extra_repr(self) -> str:
        """Return string representation with parameters."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, ceil_mode={self.ceil_mode}, count_include_pad={self.count_include_pad}"


class AvgPool2d(Module):
    """
    Average Pooling 2D layer.

    Applies average pooling operation to spatial input. Divides input into pooling
    windows and outputs the average (mean) value from each window.

    Useful for smoothing feature maps and reducing spatial dimensions while
    preserving overall statistical information.

    Example:
        >>> pool = AvgPool2d(kernel_size=2, stride=2)
        >>> x = torch.randn(batch_size, 3, 224, 224)
        >>> output = pool(x)
        >>> assert output.shape == (batch_size, 3, 112, 112)

    Attributes:
        kernel_size: Pooling window size
        stride: Step size between pooling windows
        padding: Zero-padding to apply
        ceil_mode: If True, use ceiling instead of floor for size calculation
        count_include_pad: If True, include padded positions in average calculation
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Optional[Union[int, Tuple[int, ...]]] = None,
        padding: Union[int, Tuple[int, ...]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ):
        """
        Initialize average pooling layer.

        Args:
            kernel_size: Size of pooling window (single int or tuple for h,w)
            stride: Step size between windows. If None, defaults to kernel_size
            padding: Zero-padding to apply before pooling
            ceil_mode: If True, use ceiling for output size calculation
            count_include_pad: If True, padded zeros are included in average
                             If False, average is only over unpadded values

        Raises:
            ValueError: If kernel_size <= 0
            ValueError: If stride <= 0 (when not None)
            ValueError: If padding < 0
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply average pooling to input.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Pooled tensor of shape (batch_size, channels, height_out, width_out)

        Raises:
            ValueError: If x is not 4D (batch, channels, height, width)

        Example:
            >>> pool = AvgPool2d(kernel_size=3, stride=1, padding=1)
            >>> x = torch.randn(2, 64, 28, 28)
            >>> output = pool(x)
            >>> assert output.shape == (2, 64, 28, 28)
        """
        raise NotImplementedError(
            "TODO: Implement forward pass\n"
            "1. Validate input is 4D: (batch, channels, height, width)\n"
            "2. Use torch.nn.functional.avg_pool2d with parameters:\n"
            "   F.avg_pool2d(x, kernel_size, stride, padding,\n"
            "                ceil_mode, count_include_pad)\n"
            "3. Return output"
        )

    def compute_output_size(
        self, input_size: Union[int, Tuple[int, int]]
    ) -> Union[int, Tuple[int, int]]:
        """
        Compute output spatial size given input size.

        Args:
            input_size: Input spatial size(s)

        Returns:
            Output spatial size(s)

        Formula:
            output = floor((input + 2*padding - kernel_size) / stride) + 1
            OR if ceil_mode: use ceil instead of floor

        Example:
            >>> pool = AvgPool2d(kernel_size=2, stride=2)
            >>> out_size = pool.compute_output_size(input_size=224)
            >>> assert out_size == 112
        """
        raise NotImplementedError(
            "TODO: Compute output size\n"
            "Apply pooling size formula"
        )

    def extra_repr(self) -> str:
        """Return string representation with parameters."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, ceil_mode={self.ceil_mode}, count_include_pad={self.count_include_pad}"


class AvgPool3d(Module):
    """
    Average Pooling 3D layer.

    Applies average pooling operation to volumetric input.

    Attributes:
        kernel_size: Pooling window size
        stride: Step size between pooling windows
        padding: Zero-padding to apply
        ceil_mode: If True, use ceiling instead of floor for size calculation
        count_include_pad: If True, include padded positions in average calculation
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ):
        """
        Initialize average pooling 3D layer.

        Args:
            kernel_size: Size of pooling window
            stride: Step size between windows. If None, defaults to kernel_size
            padding: Zero-padding to apply before pooling
            ceil_mode: If True, use ceiling for output size calculation
            count_include_pad: If True, padded zeros are included in average
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply average pooling to input.

        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)

        Returns:
            Pooled tensor of shape (batch_size, channels, D_out, H_out, W_out)
        """
        raise NotImplementedError(
            "TODO: Implement forward pass\n"
            "1. Validate input is 5D: (batch, channels, depth, height, width)\n"
            "2. Use avg_pool3d functional with parameters\n"
            "3. Return output"
        )

    def extra_repr(self) -> str:
        """Return string representation with parameters."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AdaptiveAvgPool1d(Module):
    """
    Adaptive Average Pooling 1D.

    Automatically computes kernel and stride to achieve desired output size.

    Attributes:
        output_size: Target output size
    """

    def __init__(self, output_size: Union[int, Tuple[int]]):
        """
        Initialize adaptive average pooling 1D.

        Args:
            output_size: Desired output size
        """
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply adaptive average pooling.

        Args:
            x: Input tensor of shape (batch, channels, length)

        Returns:
            Output tensor of shape (batch, channels, output_size)
        """
        raise NotImplementedError(
            "TODO: Implement adaptive average pooling 1D\n"
            "1. Validate input is 3D\n"
            "2. Use adaptive_avg_pool1d functional\n"
            "3. Return output"
        )


class AdaptiveAvgPool2d(Module):
    """
    Adaptive Average Pooling 2D.

    Automatically computes kernel and stride to achieve desired output size.
    Unlike AvgPool2d which specifies kernel size, AdaptiveAvgPool2d specifies
    the desired output spatial size.

    Commonly used for:
    - Image classification (global average pooling)
    - Handling variable input sizes
    - Feature compression before fully connected layers

    Example:
        >>> # Global average pooling
        >>> global_pool = AdaptiveAvgPool2d(output_size=1)
        >>> x = torch.randn(batch_size, 2048, 7, 7)
        >>> output = global_pool(x)
        >>> assert output.shape == (batch_size, 2048, 1, 1)

        >>> # Fixed output size
        >>> pool = AdaptiveAvgPool2d(output_size=(7, 7))
        >>> x = torch.randn(batch_size, channels, height, width)  # Any size
        >>> output = pool(x)
        >>> assert output.shape == (batch_size, channels, 7, 7)

    Attributes:
        output_size: Target output spatial size
    """

    def __init__(self, output_size: Union[int, Tuple[int, ...]]):
        """
        Initialize adaptive average pooling.

        Args:
            output_size: Desired output spatial size. Can be:
                - Single int: output will be (output_size, output_size)
                - Tuple of ints: output will be (h, w)

        Raises:
            ValueError: If output_size <= 0

        Note:
            Common usage:
            - output_size=1: Global average pooling
            - output_size=7: For 224x224 images -> 7x7 feature maps
            - output_size=14: For 448x448 images -> 14x14 feature maps
        """
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply adaptive average pooling.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, channels, H_out, W_out)

        Raises:
            ValueError: If x is not 4D

        Example:
            >>> pool = AdaptiveAvgPool2d(output_size=1)
            >>> x = torch.randn(32, 2048, 7, 7)
            >>> output = pool(x)
            >>> assert output.shape == (32, 2048, 1, 1)
            >>> # Can reshape to (32, 2048) for downstream FC layers
        """
        raise NotImplementedError(
            "TODO: Implement adaptive average pooling\n"
            "1. Validate input is 4D\n"
            "2. Use torch.nn.functional.adaptive_avg_pool2d:\n"
            "   F.adaptive_avg_pool2d(x, self.output_size)\n"
            "3. Return output"
        )

    def extra_repr(self) -> str:
        """Return string representation with parameters."""
        return f"output_size={self.output_size}"


class AdaptiveAvgPool3d(Module):
    """
    Adaptive Average Pooling 3D.

    Automatically computes kernel and stride to achieve desired output size
    for volumetric data.

    Attributes:
        output_size: Target output spatial size
    """

    def __init__(self, output_size: Union[int, Tuple[int, int, int]]):
        """
        Initialize adaptive average pooling 3D.

        Args:
            output_size: Desired output spatial size (D, H, W)
        """
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply adaptive average pooling.

        Args:
            x: Input tensor of shape (batch, channels, depth, height, width)

        Returns:
            Output tensor of shape (batch, channels, D_out, H_out, W_out)
        """
        raise NotImplementedError(
            "TODO: Implement adaptive average pooling 3D\n"
            "1. Validate input is 5D\n"
            "2. Use adaptive_avg_pool3d functional\n"
            "3. Return output"
        )


# ============================================================================
# Global Pooling Classes
# ============================================================================

class GlobalAvgPool1d(Module):
    """
    Global Average Pooling 1D.

    Computes mean across the temporal/spatial dimension, reducing to (batch, channels).

    Equivalent to: AdaptiveAvgPool1d(output_size=1) + squeeze

    Example:
        >>> pool = GlobalAvgPool1d()
        >>> x = torch.randn(batch_size, channels, length)
        >>> output = pool(x)
        >>> assert output.shape == (batch_size, channels)
    """

    def __init__(self, eps: float = 1e-6):
        """
        Initialize global average pooling 1D.

        Args:
            eps: Small value for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply global average pooling.

        Args:
            x: Input tensor of shape (batch_size, channels, length)

        Returns:
            Output tensor of shape (batch_size, channels)
        """
        raise NotImplementedError(
            "TODO: Implement global average pooling 1D\n"
            "1. Validate input is 3D: (batch, channels, length)\n"
            "2. Compute mean over dimension 2\n"
            "3. Return output of shape (batch, channels)"
        )


class GlobalMaxPool1d(Module):
    """
    Global Max Pooling 1D.

    Computes maximum across the temporal/spatial dimension, reducing to (batch, channels).

    Example:
        >>> pool = GlobalMaxPool1d()
        >>> x = torch.randn(batch_size, channels, length)
        >>> output = pool(x)
        >>> assert output.shape == (batch_size, channels)
    """

    def __init__(self):
        """Initialize global max pooling 1D."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply global max pooling.

        Args:
            x: Input tensor of shape (batch_size, channels, length)

        Returns:
            Output tensor of shape (batch_size, channels)
        """
        raise NotImplementedError(
            "TODO: Implement global max pooling 1D\n"
            "1. Validate input is 3D\n"
            "2. Compute max over dimension 2\n"
            "3. Return output of shape (batch, channels)"
        )


class GlobalAvgPool2d(Module):
    """
    Global Average Pooling 2D.

    Computes mean across spatial dimensions, reducing feature map to (batch, channels).

    Equivalent to: AdaptiveAvgPool2d(output_size=1) + squeeze

    Commonly used as the final pooling layer in image classification networks,
    replacing fully connected layers for parameter efficiency and better generalization.

    Example:
        >>> pool = GlobalAvgPool2d()
        >>> x = torch.randn(batch_size, 2048, 7, 7)
        >>> output = pool(x)
        >>> assert output.shape == (batch_size, 2048)

        >>> # In a network
        >>> features = backbone(images)  # (batch, 2048, 7, 7)
        >>> pooled = GlobalAvgPool2d()(features)  # (batch, 2048)
        >>> logits = classifier(pooled)  # (batch, num_classes)

    Attributes:
        eps (float): Small value for numerical stability (rarely needed)
    """

    def __init__(self, eps: float = 1e-6):
        """
        Initialize global average pooling.

        Args:
            eps: Small value for numerical stability (rarely needed)
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply global average pooling.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, channels)

        Raises:
            ValueError: If x is not 4D

        Example:
            >>> pool = GlobalAvgPool2d()
            >>> x = torch.randn(32, 2048, 7, 7)
            >>> output = pool(x)
            >>> assert output.shape == (32, 2048)
        """
        raise NotImplementedError(
            "TODO: Implement global average pooling\n"
            "1. Validate input is 4D: (batch, channels, height, width)\n"
            "2. Compute mean over spatial dimensions (dims 2, 3)\n"
            "   Option 1: x.mean(dim=(2, 3))\n"
            "   Option 2: F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)\n"
            "3. Return output of shape (batch, channels)\n"
            "\n"
            "Note: x.mean(dim=(2, 3)) is simpler and more direct"
        )


class GlobalMaxPool2d(Module):
    """
    Global Max Pooling 2D.

    Computes maximum across spatial dimensions, reducing feature map to (batch, channels).

    Equivalent to: AdaptiveMaxPool2d(output_size=1) + squeeze

    Useful for:
    - Detecting presence of specific patterns regardless of location
    - Attention mechanisms
    - Bag-of-words style features

    Less commonly used than GlobalAvgPool2d, but can be effective in certain applications.

    Example:
        >>> pool = GlobalMaxPool2d()
        >>> x = torch.randn(batch_size, 2048, 7, 7)
        >>> output = pool(x)
        >>> assert output.shape == (batch_size, 2048)
    """

    def __init__(self):
        """Initialize global max pooling."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply global max pooling.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, channels)

        Raises:
            ValueError: If x is not 4D

        Example:
            >>> pool = GlobalMaxPool2d()
            >>> x = torch.randn(32, 2048, 7, 7)
            >>> output = pool(x)
            >>> assert output.shape == (32, 2048)
        """
        raise NotImplementedError(
            "TODO: Implement global max pooling\n"
            "1. Validate input is 4D\n"
            "2. Compute max over spatial dimensions (dims 2, 3)\n"
            "   x.amax(dim=(2, 3))\n"
            "3. Return output of shape (batch, channels)"
        )


class LPPool2d(Module):
    """
    Power-average pooling layer (Lp pooling).

    Instead of max or average, computes the Lp norm over each pooling region:
        output = (sum(x^p) / kernel_elements)^(1/p)

    When p=1, this becomes average pooling.
    When p=inf, this approaches max pooling.

    Useful for learning optimal pooling behavior between average and max.

    Example:
        >>> pool = LPPool2d(norm_type=2, kernel_size=2, stride=2)
        >>> x = torch.randn(batch_size, channels, height, width)
        >>> output = pool(x)

    Attributes:
        norm_type: The exponent p in the Lp norm
        kernel_size: Pooling window size
        stride: Step size between pooling windows
        ceil_mode: If True, use ceiling for output size calculation
    """

    def __init__(
        self,
        norm_type: float,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        ceil_mode: bool = False,
    ):
        """
        Initialize LP pooling layer.

        Args:
            norm_type: The exponent p (typically 2 for L2 pooling)
            kernel_size: Size of pooling window
            stride: Step size between windows. If None, defaults to kernel_size
            ceil_mode: If True, use ceiling for output size calculation
        """
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply LP pooling to input.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Pooled tensor of shape (batch_size, channels, height_out, width_out)
        """
        raise NotImplementedError(
            "TODO: Implement LP pooling forward pass\n"
            "1. Validate input is 4D\n"
            "2. For each pooling window:\n"
            "   - Compute abs(x)^p\n"
            "   - Sum over window\n"
            "   - Take p-th root: (sum)^(1/p)\n"
            "3. Return pooled output"
        )

    def extra_repr(self) -> str:
        """Return string representation with parameters."""
        return f"norm_type={self.norm_type}, kernel_size={self.kernel_size}, stride={self.stride}"


class MaxUnpool2d(Module):
    """
    Max Unpooling 2D layer (inverse of MaxPool2d).

    Uses indices from MaxPool2d to place values back in their original
    positions during the "unpooling" operation. Commonly used in
    encoder-decoder architectures like SegNet.

    Example:
        >>> pool = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        >>> unpool = MaxUnpool2d(kernel_size=2, stride=2)
        >>> x = torch.randn(batch_size, channels, height, width)
        >>> output, indices = pool(x)
        >>> unpooled = unpool(output, indices)

    Attributes:
        kernel_size: Pooling window size (must match corresponding MaxPool2d)
        stride: Step size (must match corresponding MaxPool2d)
        padding: Padding (must match corresponding MaxPool2d)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        """
        Initialize max unpooling layer.

        Args:
            kernel_size: Size of pooling window from MaxPool2d
            stride: Stride from MaxPool2d. If None, defaults to kernel_size
            padding: Padding from MaxPool2d
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(
        self,
        x: Tensor,
        indices: Tensor,
        output_size: Optional[Tuple[int, ...]] = None,
    ) -> Tensor:
        """
        Apply max unpooling.

        Args:
            x: Input tensor from MaxPool2d output
            indices: Indices tensor from MaxPool2d (positions of max values)
            output_size: Optional explicit output size

        Returns:
            Unpooled tensor with values placed at original max positions
        """
        raise NotImplementedError(
            "TODO: Implement max unpooling forward pass\n"
            "1. Determine output size (from output_size arg or computed)\n"
            "2. Create output tensor filled with zeros\n"
            "3. Use indices to place input values in output\n"
            "4. Return unpooled tensor"
        )

    def extra_repr(self) -> str:
        """Return string representation with parameters."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class MultiHeadGlobalPooling(Module):
    """
    Multi-head global pooling with multiple aggregation methods.

    Computes global pooling using multiple strategies and concatenates results.
    Allows the network to learn from both average and max information.

    Strategies can include:
    - Global average pooling
    - Global max pooling
    - Global min pooling
    - Global standard deviation
    - Global median

    Example:
        >>> pool = MultiHeadGlobalPooling(strategies=['avg', 'max'])
        >>> x = torch.randn(batch_size, 2048, 7, 7)
        >>> output = pool(x)
        >>> assert output.shape == (batch_size, 2048 * 2)  # Doubled channels
    """

    def __init__(
        self,
        strategies: List[str] = None,
        keep_dims: bool = False,
    ):
        """
        Initialize multi-head global pooling.

        Args:
            strategies: List of pooling strategies to use.
                       Can include: 'avg', 'max', 'min', 'std'
                       Default: ['avg', 'max']
            keep_dims: If True, keep spatial dimensions as 1
                      If False, squeeze to (batch, channels)

        Raises:
            ValueError: If strategies is empty
            ValueError: If unknown strategy name
        """
        super().__init__()
        self.strategies = strategies if strategies is not None else ['avg', 'max']
        self.keep_dims = keep_dims

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply multi-head global pooling.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Concatenated output of all strategies
            Shape: (batch, channels * num_strategies) or (batch, channels * num_strategies, 1, 1)

        Example:
            >>> pool = MultiHeadGlobalPooling(strategies=['avg', 'max'])
            >>> x = torch.randn(32, 2048, 7, 7)
            >>> output = pool(x)
            >>> assert output.shape == (32, 2048 * 2)
        """
        raise NotImplementedError(
            "TODO: Implement multi-head global pooling\n"
            "1. For each strategy, compute the pooling\n"
            "2. Stack/concatenate results along channel dimension\n"
            "3. If keep_dims=True, reshape to (batch, channels*num_strategies, 1, 1)\n"
            "4. If keep_dims=False, reshape to (batch, channels*num_strategies)\n"
            "5. Return result"
        )

    @staticmethod
    def _pool_strategy(
        x: Tensor,
        strategy: str,
        keep_dims: bool = False,
    ) -> Tensor:
        """
        Apply a single pooling strategy.

        Args:
            x: Input tensor
            strategy: Pooling strategy name
            keep_dims: Whether to keep spatial dimensions

        Returns:
            Pooled tensor
        """
        raise NotImplementedError(
            "TODO: Implement pooling strategy dispatcher\n"
            "- 'avg': x.mean(dim=(2, 3))\n"
            "- 'max': x.amax(dim=(2, 3))\n"
            "- 'min': x.amin(dim=(2, 3))\n"
            "- 'std': x.std(dim=(2, 3))\n"
            "- Add keepdim=keep_dims to preserve spatial dims if needed"
        )


class AdaptivePooling(Module):
    """
    Flexible adaptive pooling with configurable output size and method.

    Combines adaptive max and average pooling with automatic method selection.

    Example:
        >>> pool = AdaptivePooling(output_size=1, method='avg')
        >>> x = torch.randn(batch_size, channels, height, width)
        >>> output = pool(x)
    """

    def __init__(
        self,
        output_size: Union[int, tuple] = 1,
        method: str = "avg",
    ):
        """
        Initialize adaptive pooling.

        Args:
            output_size: Target output spatial size
            method: 'avg' or 'max'

        Raises:
            ValueError: If method not in ['avg', 'max']
        """
        super().__init__()
        self.output_size = output_size
        self.method = method

    def forward(self, x: Tensor) -> Tensor:
        """Apply adaptive pooling."""
        raise NotImplementedError(
            "TODO: Implement adaptive pooling forward\n"
            "Use the appropriate pooling layer based on self.method"
        )


# ============================================================================
# Spatial Pyramid Pooling
# ============================================================================

class SpatialPyramidPooling(Module):
    """
    Spatial Pyramid Pooling (SPP).

    Applies multiple adaptive pooling operations at different scales
    and concatenates results, creating a multi-scale representation.

    This creates an output of fixed size regardless of input size
    by computing features at multiple pooling levels.

    Formula:
    For levels [L1, L2, L3]:
    - L1: 1×1 pooling (global pooling)
    - L2: 2×2 pooling
    - L3: 4×4 pooling
    Output: concatenate all level outputs

    Example:
        >>> spp = SpatialPyramidPooling(levels=[1, 2, 4])
        >>> x = torch.randn(32, 256, height, width)  # Any size
        >>> output = spp(x)
        >>> # Output has num_channels * (1 + 4 + 16) = num_channels * 21 channels
        >>> expected_channels = 256 * (1 + 4 + 16)
        >>> assert output.shape[1] == expected_channels

    PAPER REFERENCE:
    "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition"
    - He et al., 2014
    https://arxiv.org/abs/1406.4729

    Attributes:
        levels: List of pyramid levels (e.g., [1, 2, 4])
        pooling_method: 'avg' or 'max'
    """

    def __init__(
        self,
        levels: List[int] = None,
        pooling_method: str = "max",
    ):
        """
        Initialize spatial pyramid pooling.

        Args:
            levels: Pyramid levels. Default [1, 2, 4] creates 3 levels
                   Level k means k×k pooling
            pooling_method: 'avg' or 'max' pooling

        Raises:
            ValueError: If levels is empty
            ValueError: If any level <= 0
        """
        super().__init__()
        self.levels = levels if levels is not None else [1, 2, 4]
        self.pooling_method = pooling_method

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply spatial pyramid pooling.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Concatenated pyramid features
            Shape: (batch, channels * total_positions, 1, 1)
            Where total_positions = sum(level^2 for level in levels)

        Example:
            >>> spp = SpatialPyramidPooling(levels=[1, 2, 4])
            >>> x = torch.randn(32, 256, 13, 13)
            >>> output = spp(x)
            >>> # 1^2 + 2^2 + 4^2 = 1 + 4 + 16 = 21
            >>> assert output.shape == (32, 256 * 21, 1, 1)
        """
        raise NotImplementedError(
            "TODO: Implement spatial pyramid pooling\n"
            "1. Apply adaptive pooling for each level\n"
            "2. Flatten spatial dimensions for each level\n"
            "3. Concatenate all level outputs along channel dimension\n"
            "4. Return concatenated result"
        )

    def compute_output_channels(self, input_channels: int) -> int:
        """
        Compute output channel count after SPP.

        Args:
            input_channels: Number of input channels

        Returns:
            Number of output channels after concatenating all levels
        """
        raise NotImplementedError(
            "TODO: Compute output channels\n"
            "return input_channels * sum(level**2 for level in self.levels)"
        )


# ============================================================================
# Advanced Pooling Variants
# ============================================================================

class MixedPooling(Module):
    """
    Mixed pooling combining max and average.

    Some research suggests combining max and average pooling can be beneficial:
    - Max captures peaks (edges, corners)
    - Average captures overall statistics (colors, textures)
    - Combination captures both

    Formula:
        output = alpha * max_pool + (1 - alpha) * avg_pool

    Example:
        >>> pool = MixedPooling(kernel_size=2, alpha=0.5)
        >>> x = torch.randn(batch, channels, height, width)
        >>> output = pool(x)

    REFERENCE:
    "Mixed Pooling for Convolutional Neural Networks"
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Optional[Union[int, Tuple[int, ...]]] = None,
        padding: Union[int, Tuple[int, ...]] = 0,
        alpha: float = 0.5,
    ):
        """
        Initialize mixed pooling layer.

        Args:
            kernel_size: Pooling window size
            stride: Stride (defaults to kernel_size)
            padding: Zero-padding
            alpha: Weight of max pooling (0.5 = equal mix)
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply mixed pooling.

        Args:
            x: Input tensor

        Returns:
            Mixed pooled output
        """
        raise NotImplementedError(
            "TODO: Implement mixed pooling\n"
            "Apply both max and avg pooling, then blend results"
        )

    def extra_repr(self) -> str:
        """Return string representation with parameters."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, alpha={self.alpha}"


class StochasticAdaptivePooling(Module):
    """
    Stochastic Adaptive Pooling.

    Instead of deterministically selecting max/average, randomly samples
    from each pooling region with probability proportional to activation values.

    This provides regularization effect during training while using
    max/avg pooling during evaluation.

    Advantages:
    - Regularization: Reduces overfitting by stochasticity
    - Gradient diffusion: Gradients reach all positions (unlike max)
    - Trainability: Better gradient flow than deterministic max

    Example:
        >>> pool = StochasticAdaptivePooling(output_size=7)
        >>> x = torch.randn(32, 256, height, width)
        >>> output = pool(x)
        >>> assert output.shape == (32, 256, 7, 7)

    REFERENCE:
    "Fractional Max-Pooling"
    - Graham, 2014
    https://arxiv.org/abs/1412.6071
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[int, ...]],
        use_stochastic: bool = True,
        temperature: float = 1.0,
    ):
        """
        Initialize stochastic adaptive pooling.

        Args:
            output_size: Target output size
            use_stochastic: If True, use stochastic sampling in training
            temperature: Temperature for softmax over pool activations
        """
        super().__init__()
        self.output_size = output_size
        self.use_stochastic = use_stochastic
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply stochastic adaptive pooling.

        Args:
            x: Input tensor

        Returns:
            Pooled output
        """
        raise NotImplementedError(
            "TODO: Implement stochastic pooling\n"
            "During training: sample with probability proportional to activation\n"
            "During eval: use max pooling (deterministic)"
        )


# ============================================================================
# Functional Interfaces
# ============================================================================

def max_pool1d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int]],
    stride: Optional[Union[int, Tuple[int]]] = None,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Functional interface for max pooling 1D.

    Args:
        input: Input tensor of shape (batch, channels, length)
        kernel_size: Pooling window size
        stride: Stride (defaults to kernel_size)
        padding: Zero-padding
        dilation: Dilation factor
        ceil_mode: Use ceiling for size calculation
        return_indices: Return indices of max values

    Returns:
        Pooled tensor, or (pooled, indices) if return_indices=True
    """
    raise NotImplementedError(
        "TODO: Functional max pooling 1D wrapper\n"
        "Use torch.nn.functional.max_pool1d directly"
    )


def max_pool2d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Functional interface for max pooling 2D.

    Args:
        input: Input tensor of shape (batch, channels, height, width)
        kernel_size: Pooling window size
        stride: Stride (defaults to kernel_size)
        padding: Zero-padding
        dilation: Dilation factor
        ceil_mode: Use ceiling for size calculation
        return_indices: Return indices of max values

    Returns:
        Pooled tensor, or (pooled, indices) if return_indices=True
    """
    raise NotImplementedError(
        "TODO: Functional max pooling 2D wrapper\n"
        "Use torch.nn.functional.max_pool2d directly"
    )


def avg_pool1d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int]],
    stride: Optional[Union[int, Tuple[int]]] = None,
    padding: Union[int, Tuple[int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """
    Functional interface for average pooling 1D.

    Args:
        input: Input tensor of shape (batch, channels, length)
        kernel_size: Pooling window size
        stride: Stride (defaults to kernel_size)
        padding: Zero-padding
        ceil_mode: Use ceiling for size calculation
        count_include_pad: Include padded positions in average

    Returns:
        Pooled tensor
    """
    raise NotImplementedError(
        "TODO: Functional average pooling 1D wrapper\n"
        "Use torch.nn.functional.avg_pool1d directly"
    )


def avg_pool2d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """
    Functional interface for average pooling 2D.

    Args:
        input: Input tensor of shape (batch, channels, height, width)
        kernel_size: Pooling window size
        stride: Stride (defaults to kernel_size)
        padding: Zero-padding
        ceil_mode: Use ceiling for size calculation
        count_include_pad: Include padded positions in average

    Returns:
        Pooled tensor
    """
    raise NotImplementedError(
        "TODO: Functional average pooling 2D wrapper\n"
        "Use torch.nn.functional.avg_pool2d directly"
    )


def adaptive_max_pool2d(
    x: Tensor,
    output_size: Union[int, Tuple[int, int]],
    return_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Functional interface for adaptive max pooling.

    Args:
        x: Input tensor of shape (batch, channels, height, width)
        output_size: Target output spatial size
        return_indices: Return indices of max values

    Returns:
        Pooled tensor, or (pooled, indices) if return_indices=True
    """
    raise NotImplementedError(
        "TODO: Functional adaptive max pooling\n"
        "Use F.adaptive_max_pool2d directly"
    )


def adaptive_avg_pool2d(
    x: Tensor,
    output_size: Union[int, Tuple[int, int]],
) -> Tensor:
    """
    Functional interface for adaptive average pooling.

    Args:
        x: Input tensor of shape (batch, channels, height, width)
        output_size: Target output spatial size

    Returns:
        Pooled tensor
    """
    raise NotImplementedError(
        "TODO: Functional adaptive avg pooling\n"
        "Use F.adaptive_avg_pool2d directly"
    )


def global_avg_pool2d(x: Tensor) -> Tensor:
    """
    Apply global average pooling (average over spatial dimensions).

    Reduces spatial dimensions to 1x1 while preserving channels.

    Args:
        x: Input tensor of shape (batch, channels, height, width)

    Returns:
        Output tensor of shape (batch, channels, 1, 1)

    Example:
        >>> x = torch.randn(32, 2048, 7, 7)
        >>> output = global_avg_pool2d(x)
        >>> assert output.shape == (32, 2048, 1, 1)

    Note:
        Equivalent to:
        - F.adaptive_avg_pool2d(x, 1)
        - x.mean(dim=(-2, -1), keepdim=True)
    """
    raise NotImplementedError(
        "TODO: Implement global average pooling\n"
        "1. Use F.adaptive_avg_pool2d(x, 1)\n"
        "2. Return result"
    )


def global_max_pool2d(x: Tensor) -> Tensor:
    """
    Apply global max pooling (max over spatial dimensions).

    Reduces spatial dimensions to 1x1 while preserving channels.

    Args:
        x: Input tensor of shape (batch, channels, height, width)

    Returns:
        Output tensor of shape (batch, channels, 1, 1)

    Example:
        >>> x = torch.randn(32, 2048, 7, 7)
        >>> output = global_max_pool2d(x)
        >>> assert output.shape == (32, 2048, 1, 1)

    Note:
        Equivalent to:
        - F.adaptive_max_pool2d(x, 1)
        - x.amax(dim=(-2, -1), keepdim=True)
    """
    raise NotImplementedError(
        "TODO: Implement global max pooling\n"
        "1. Use F.adaptive_max_pool2d(x, 1)\n"
        "2. Return result"
    )


def global_pool2d(
    x: Tensor,
    method: str = "avg",
) -> Tensor:
    """
    Functional interface for flexible global pooling.

    Args:
        x: Input tensor of shape (batch, channels, height, width)
        method: 'avg' or 'max'

    Returns:
        Output tensor of shape (batch, channels, 1, 1)
    """
    raise NotImplementedError(
        "TODO: Implement flexible global pooling\n"
        "Dispatch to global_avg_pool2d or global_max_pool2d based on method"
    )


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

    Args:
        input_size: Input spatial size
        kernel_size: Pooling window size
        stride: Stride between windows
        padding: Zero-padding
        dilation: Kernel dilation
        ceil_mode: Use ceiling or floor

    Returns:
        Output spatial size

    Formula:
        output = floor((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
    """
    raise NotImplementedError(
        "TODO: Compute output size\n"
        "Implement pooling size formula"
    )
