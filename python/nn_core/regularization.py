"""
nn_core.regularization - Regularization Modules and Functions

This module provides regularization operations for deep learning, including:
- Dropout: Standard element-wise dropout
- Dropout1d/2d/3d: Spatial dropout for convolutional networks
- DropPath: Stochastic depth for residual networks
- LabelSmoothing: Label smoothing for classification
- Functional classes: DropoutFunction, Dropout2dFunction, DropPathFunction, etc.

All regularization modules should be used with forward() method and autograd handles backward.
"""

from typing import Tuple, Optional, Union
import numpy as np
from python.foundations import Tensor
from .module import Module, Parameter
from python.foundations.functionals import Function


# ============================================================================
# Functional Classes (Autograd Functions)
# ============================================================================

class DropoutFunction(Function):
    """
    Dropout regularization functional operation.

    Randomly sets elements to zero during training with probability p.
    Scales remaining elements by 1/(1-p) to maintain expected values.

    Math:
        Training:
            mask ~ Bernoulli(1 - p)
            y = x * mask / (1 - p)

        Inference:
            y = x

    Note: Uses "inverted dropout" where scaling happens at training time,
    so no changes needed at inference.

    Example:
        >>> fn = DropoutFunction(p=0.5)
        >>> x = Tensor(np.random.randn(32, 128))
        >>> y = fn.forward(x, training=True)
        >>> # About 50% of elements are zero, rest are scaled by 2
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize Dropout function.

        Args:
            p: Probability of dropping an element (0 to 1)
        """
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """
        Apply dropout.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor with dropout applied (if training)
        """
        raise NotImplementedError(
            "Implement Dropout forward:\n"
            "  self.training = training\n"
            "  \n"
            "  if not training or self.p == 0:\n"
            "      return x\n"
            "  \n"
            "  # Generate dropout mask\n"
            "  self.mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)\n"
            "  \n"
            "  # Apply mask and scale\n"
            "  return x * self.mask / (1 - self.p)"
        )

    def backward(self, grad_output: Tensor) -> Tuple[Tensor]:
        """
        Compute gradient for dropout.

        Args:
            grad_output: Gradient w.r.t. output

        Returns:
            Tuple of (grad_input,)
        """
        raise NotImplementedError(
            "Implement Dropout backward:\n"
            "  if not self.training or self.p == 0:\n"
            "      return (grad_output,)\n"
            "  \n"
            "  # Gradient flows only through kept elements\n"
            "  return (grad_output * self.mask / (1 - self.p),)"
        )


class Dropout2dFunction(Function):
    """
    2D Spatial Dropout functional operation.

    Drops entire channels (feature maps) rather than individual elements.
    More suitable for convolutional layers where spatial features are correlated.

    Input shape: (batch, channels, height, width)
    Drops channels with probability p.

    Example:
        >>> fn = Dropout2dFunction(p=0.2)
        >>> x = Tensor(np.random.randn(8, 64, 32, 32))
        >>> y = fn.forward(x, training=True)
        >>> # About 20% of channels (all spatial positions) are zero
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize Dropout2d function.

        Args:
            p: Probability of dropping a channel
        """
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """
        Apply 2D spatial dropout.

        Args:
            x: Input tensor (batch, channels, height, width)
            training: Whether in training mode

        Returns:
            Output tensor
        """
        raise NotImplementedError(
            "Implement Dropout2d forward:\n"
            "  self.training = training\n"
            "  \n"
            "  if not training or self.p == 0:\n"
            "      return x\n"
            "  \n"
            "  batch, channels, h, w = x.shape\n"
            "  \n"
            "  # Mask shape: (batch, channels, 1, 1) - same mask for all spatial positions\n"
            "  self.mask = (np.random.rand(batch, channels, 1, 1) > self.p).astype(x.dtype)\n"
            "  \n"
            "  return x * self.mask / (1 - self.p)"
        )

    def backward(self, grad_output: Tensor) -> Tuple[Tensor]:
        """
        Compute gradient for 2D spatial dropout.

        Args:
            grad_output: Gradient w.r.t. output

        Returns:
            Tuple of (grad_input,)
        """
        raise NotImplementedError(
            "Implement Dropout2d backward:\n"
            "  if not self.training or self.p == 0:\n"
            "      return (grad_output,)\n"
            "  \n"
            "  return (grad_output * self.mask / (1 - self.p),)"
        )


class DropPathFunction(Function):
    """
    Drop Path (Stochastic Depth) functional operation.

    Drops entire samples (residual branches) during training.
    Used in residual networks to regularize and improve training.

    Unlike Dropout which drops elements, DropPath drops the entire
    contribution of a residual branch for selected samples.

    Math:
        Training:
            For each sample i: y_i = x_i * mask_i / (1 - p)
            where mask_i ~ Bernoulli(1 - p)

        Inference:
            y = x

    Example:
        >>> fn = DropPathFunction(drop_prob=0.1)
        >>> residual = Tensor(np.random.randn(32, 64, 8, 8))
        >>> output = fn.forward(residual, training=True)
        >>> # About 10% of samples have residual = 0

    References:
        - Deep Networks with Stochastic Depth (Huang et al., 2016)
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Initialize DropPath function.

        Args:
            drop_prob: Probability of dropping a sample's path
        """
        self.drop_prob = drop_prob

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """
        Apply drop path.

        Args:
            x: Input tensor (batch, ...)
            training: Whether in training mode

        Returns:
            Output tensor
        """
        raise NotImplementedError(
            "Implement DropPath forward:\n"
            "  self.training = training\n"
            "  \n"
            "  if not training or self.drop_prob == 0:\n"
            "      return x\n"
            "  \n"
            "  # Keep probability\n"
            "  keep_prob = 1 - self.drop_prob\n"
            "  \n"
            "  # Mask shape: (batch, 1, 1, ...) - one value per sample\n"
            "  shape = (x.shape[0],) + (1,) * (x.ndim - 1)\n"
            "  self.mask = (np.random.rand(*shape) < keep_prob).astype(x.dtype)\n"
            "  \n"
            "  return x * self.mask / keep_prob"
        )

    def backward(self, grad_output: Tensor) -> Tuple[Tensor]:
        """
        Compute gradient for drop path.

        Args:
            grad_output: Gradient w.r.t. output

        Returns:
            Tuple of (grad_input,)
        """
        raise NotImplementedError(
            "Implement DropPath backward:\n"
            "  if not self.training or self.drop_prob == 0:\n"
            "      return (grad_output,)\n"
            "  \n"
            "  keep_prob = 1 - self.drop_prob\n"
            "  return (grad_output * self.mask / keep_prob,)"
        )


class DropBlockFunction(Function):
    """
    DropBlock regularization functional operation.

    Drops contiguous regions (blocks) of feature maps.
    More effective than standard dropout for convolutional networks
    because it removes semantic information that could be reconstructed
    from neighboring activations.

    Input shape: (batch, channels, height, width)

    Math:
        1. Sample mask centers with probability gamma
        2. Expand each center to a block_size x block_size region
        3. Apply mask and rescale

        gamma is computed to achieve target drop probability:
        gamma = drop_prob * (H * W) / (block_size^2 * (H - block_size + 1) * (W - block_size + 1))

    Example:
        >>> fn = DropBlockFunction(drop_prob=0.1, block_size=7)
        >>> x = Tensor(np.random.randn(8, 256, 14, 14))
        >>> y = fn.forward(x, training=True)

    References:
        - DropBlock (Ghiasi et al., 2018): https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob: float = 0.1, block_size: int = 7):
        """
        Initialize DropBlock function.

        Args:
            drop_prob: Target probability of dropping a unit
            block_size: Size of block to drop
        """
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """
        Apply DropBlock.

        Args:
            x: Input tensor (batch, channels, height, width)
            training: Whether in training mode

        Returns:
            Output tensor
        """
        raise NotImplementedError(
            "Implement DropBlock forward:\n"
            "  self.training = training\n"
            "  \n"
            "  if not training or self.drop_prob == 0:\n"
            "      return x\n"
            "  \n"
            "  batch, channels, h, w = x.shape\n"
            "  \n"
            "  # Compute gamma (probability of sampling block center)\n"
            "  valid_h = h - self.block_size + 1\n"
            "  valid_w = w - self.block_size + 1\n"
            "  gamma = (self.drop_prob * h * w) / (self.block_size ** 2 * valid_h * valid_w)\n"
            "  \n"
            "  # Sample block centers\n"
            "  centers = np.random.rand(batch, channels, valid_h, valid_w) < gamma\n"
            "  \n"
            "  # Expand centers to blocks\n"
            "  mask = np.ones((batch, channels, h, w), dtype=x.dtype)\n"
            "  for i in range(self.block_size):\n"
            "      for j in range(self.block_size):\n"
            "          mask[:, :, i:i+valid_h, j:j+valid_w] *= ~centers\n"
            "  \n"
            "  self.mask = mask\n"
            "  \n"
            "  # Normalize by fraction of kept elements\n"
            "  count = mask.sum()\n"
            "  count_ones = mask.size\n"
            "  \n"
            "  if count > 0:\n"
            "      return x * mask * (count_ones / count)\n"
            "  return x * mask"
        )

    def backward(self, grad_output: Tensor) -> Tuple[Tensor]:
        """
        Compute gradient for DropBlock.

        Args:
            grad_output: Gradient w.r.t. output

        Returns:
            Tuple of (grad_input,)
        """
        raise NotImplementedError(
            "Implement DropBlock backward:\n"
            "  if not self.training or self.drop_prob == 0:\n"
            "      return (grad_output,)\n"
            "  \n"
            "  count = self.mask.sum()\n"
            "  count_ones = self.mask.size\n"
            "  \n"
            "  if count > 0:\n"
            "      return (grad_output * self.mask * (count_ones / count),)\n"
            "  return (grad_output * self.mask,)"
        )


class AlphaDropoutFunction(Function):
    """
    Alpha Dropout for Self-Normalizing Neural Networks (SNNs).

    Unlike standard dropout, Alpha Dropout maintains mean and variance
    of activations, making it suitable for networks using SELU activation.

    Math:
        Dropped values are set to saturation point α' (not zero)
        Remaining values are scaled and shifted to maintain statistics:
        y = a * (x * mask + α' * (1 - mask)) + b

        where a and b are computed to preserve mean and variance.

    Example:
        >>> fn = AlphaDropoutFunction(p=0.1)
        >>> x = Tensor(selu(linear(inputs)))  # SELU-activated
        >>> y = fn.forward(x, training=True)

    References:
        - Self-Normalizing Neural Networks (Klambauer et al., 2017)
          https://arxiv.org/abs/1706.02515
    """

    # SELU parameters
    ALPHA = 1.6732632423543772848170429916717
    SCALE = 1.0507009873554804934193349852946

    def __init__(self, p: float = 0.5):
        """
        Initialize AlphaDropout function.

        Args:
            p: Probability of dropping an element
        """
        self.p = p
        # Saturation point for negative inputs under SELU
        self.alpha_prime = -self.SCALE * self.ALPHA

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """
        Apply alpha dropout.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        raise NotImplementedError(
            "Implement AlphaDropout forward:\n"
            "  self.training = training\n"
            "  \n"
            "  if not training or self.p == 0:\n"
            "      return x\n"
            "  \n"
            "  # Keep probability\n"
            "  q = 1 - self.p\n"
            "  \n"
            "  # Compute a and b to maintain mean/variance\n"
            "  a = (q + self.alpha_prime ** 2 * self.p * q) ** (-0.5)\n"
            "  b = -a * (self.p * self.alpha_prime)\n"
            "  \n"
            "  self.a = a\n"
            "  self.b = b\n"
            "  \n"
            "  # Generate mask\n"
            "  self.mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)\n"
            "  \n"
            "  # Apply: kept elements stay, dropped become alpha'\n"
            "  y = x * self.mask + self.alpha_prime * (1 - self.mask)\n"
            "  \n"
            "  # Scale and shift\n"
            "  return a * y + b"
        )

    def backward(self, grad_output: Tensor) -> Tuple[Tensor]:
        """
        Compute gradient for alpha dropout.

        Args:
            grad_output: Gradient w.r.t. output

        Returns:
            Tuple of (grad_input,)
        """
        raise NotImplementedError(
            "Implement AlphaDropout backward:\n"
            "  if not self.training or self.p == 0:\n"
            "      return (grad_output,)\n"
            "  \n"
            "  # Gradient only flows through kept elements\n"
            "  return (grad_output * self.a * self.mask,)"
        )


# ============================================================================
# Module Classes (Neural Network Layers)
# ============================================================================

class Dropout(Module):
    """
    Standard Dropout Layer.

    Randomly zeros activations during training to prevent co-adaptation
    and reduce overfitting. Uses inverted dropout (scales during training).

    Shapes:
    - Input: (*, N) where N is any size
    - Output: same shape as input

    Parameters:
    -----------
    p: float
        Probability of dropping each activation (default: 0.5)
        Valid range: [0, 1)
        p=0: No dropout (identity)
        p=0.5: Drop 50% of activations
        p=0.9: Drop 90% (aggressive)

    inplace: bool
        Whether to modify input in-place (default: False)
        If True, saves memory but modifies input tensor

    Learnable Parameters:
    --------------------
    None - Dropout has no learnable parameters

    Properties:
    - Different behavior in training vs evaluation
    - No parameters to optimize
    - Independent of batch size
    - Works with any input shape

    Key Methods:
    - forward(x): Apply dropout during training, identity during eval
    - train(): Enable dropout
    - eval(): Disable dropout
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Initialize Dropout layer.

        Args:
            p: Dropout probability (probability of dropping)
            inplace: Whether to modify input in-place

        Raises:
            ValueError: If p not in [0, 1)
        """
        super().__init__()

        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")

        self.p = p
        self.inplace = inplace
        self.scale = 1.0 / (1.0 - self.p) if self.p > 0 else 1.0

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply dropout to input.

        Args:
            x: Input tensor of any shape

        Returns:
            y: Output tensor (same shape as input)
               During training: Some activations zeroed, others scaled
               During eval: Identity (x unchanged)

        Algorithm:
        1. Check if training mode:
           - If not training: return x unchanged
        2. If training and p > 0:
           a. Generate random mask: M ~ Bernoulli(1-p)
              Shape same as x, values are 0 or 1
           b. Apply mask: y = x * M
           c. Scale: y = y * (1/(1-p))
        3. Return output

        Implementation Notes:
        - Use np.random.binomial(1, 1-p, size=x.shape) for Bernoulli
        - Or use np.random.rand(x.shape) < (1-p) for bool mask
        - Cache mask for backward pass if needed
        """
        raise NotImplementedError(
            "Implement Dropout forward pass:\n"
            "1. If not self.training or self.p == 0: return x unchanged\n"
            "2. Generate binary mask from Bernoulli(1-p):\n"
            "   mask = np.random.binomial(1, 1-self.p, size=x.shape)\n"
            "   OR\n"
            "   mask = (np.random.rand(*x.shape) < (1-self.p)).astype(x.dtype)\n"
            "3. Apply mask and scale:\n"
            "   if self.inplace:\n"
            "       x *= mask * self.scale\n"
            "       return x\n"
            "   else:\n"
            "       return x * mask * self.scale\n"
            "4. Cache mask for backward pass (if implementing grad)"
        )

    def extra_repr(self) -> str:
        """Return extra representation with dropout probability info."""
        return f"p={self.p}"

    def train(self):
        """Enable training mode (activate dropout)."""
        self.training = True
        return self

    def eval(self):
        """Enable evaluation mode (deactivate dropout)."""
        self.training = False
        return self


class Dropout1d(Module):
    """
    Spatial Dropout for 1D Convolutional Networks (sequence data).

    Drops entire feature maps across temporal dimension.

    Shapes:
    - Input: (N, C, L) for Conv1d output (sequences, time series)
    - Output: (N, C, L)

    Parameters:
    -----------
    p: float
        Probability of dropping each channel

    inplace: bool
        Whether to modify input in-place (default: False)

    Usage for sequence processing:
    ```
    class Conv1dBlock(Module):
        def __init__(self, in_channels, out_channels):
            self.conv = Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            self.dropout = Dropout1d(p=0.5)

        def forward(self, x):  # x: (N, C, L) sequence length
            x = self.conv(x)
            x = self.dropout(x)
            return x
    ```

    This is also useful for RNNs:
    - Can be applied to hidden states
    - Same mask across time steps
    - Preserves temporal patterns
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Initialize Dropout1d.

        Args:
            p: Dropout probability
            inplace: Whether to modify input in-place
        """
        super().__init__()

        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")

        self.p = p
        self.inplace = inplace
        self.scale = 1.0 / (1.0 - self.p) if self.p > 0 else 1.0

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply spatial dropout to 1D input.

        Args:
            x: Input tensor of shape (N, C, L)

        Returns:
            y: Output with dropped channels

        Algorithm:
        - Same as Dropout2d but for 3D input (N, C, L)
        - Mask shape: (N, C, 1)
        - Broadcasts across L (sequence length)
        """
        raise NotImplementedError(
            "Implement Dropout1d forward:\n"
            "1. Validate input shape is 3D: (N, C, L)\n"
            "2. If not self.training or self.p == 0: return x\n"
            "3. Generate channel mask (N, C)\n"
            "4. Reshape to (N, C, 1) for broadcasting\n"
            "5. Apply mask and scale\n"
            "6. Return output"
        )

    def extra_repr(self) -> str:
        """Return extra representation with dropout probability info."""
        return f"p={self.p}"

    def train(self):
        """Enable training mode."""
        self.training = True
        return self

    def eval(self):
        """Enable evaluation mode."""
        self.training = False
        return self


class Dropout2d(Module):
    """
    Spatial Dropout for 2D Convolutional Networks.

    Drops entire feature maps (channels) instead of individual activations,
    preserving spatial structure. Recommended for use after Conv2d layers.

    Shapes:
    - Input: (N, C, H, W) for Conv2d output
    - Output: (N, C, H, W)

    Parameters:
    -----------
    p: float
        Probability of dropping each channel (default: 0.5)
        All spatial positions of a dropped channel are set to 0

    inplace: bool
        Whether to modify input in-place (default: False)

    Properties:
    - Different behavior in training vs evaluation
    - Drops same mask across spatial dimensions
    - Better than standard Dropout for convolutional features
    - Preserves learned spatial patterns

    Key Difference from Dropout:
    - Dropout: Drops individual elements (shape N, C, H, W)
    - Dropout2d: Drops entire channels (mask shape N, C, 1, 1)

    Typical Usage:
    ```
    class ConvBlock(Module):
        def __init__(self, in_channels, out_channels):
            self.conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn = BatchNorm2d(out_channels)
            self.dropout = Dropout2d(p=0.5)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = relu(x)
            x = self.dropout(x)  # Drop entire feature maps
            return x
    ```
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Initialize Dropout2d.

        Args:
            p: Dropout probability (channel drop probability)
            inplace: Whether to modify input in-place

        Raises:
            ValueError: If p not in [0, 1)
        """
        super().__init__()

        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")

        self.p = p
        self.inplace = inplace
        self.scale = 1.0 / (1.0 - self.p) if self.p > 0 else 1.0

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply spatial dropout to 2D input.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            y: Output tensor (same shape as input)
               Entire channels randomly dropped during training

        Algorithm:
        1. Check training mode and p value
        2. If training and p > 0:
           a. Get batch and channel dimensions: N = x.shape[0], C = x.shape[1]
           b. Generate channel mask: M ∈ {0,1}^(N, C)
              M = Bernoulli(1-p) for each (n,c) pair
           c. Reshape mask to (N, C, 1, 1) for broadcasting
           d. Apply mask and scale: y = x * M / (1-p)
        3. Otherwise return x unchanged

        Implementation Notes:
        - Input must be 4D (N, C, H, W)
        - Mask only varies in channel dim
        - Same mask across all spatial positions (H, W)
        - During eval: return input unchanged
        """
        raise NotImplementedError(
            "Implement Dropout2d forward pass:\n"
            "1. Validate input shape is 4D: (N, C, H, W)\n"
            "2. If not self.training or self.p == 0: return x unchanged\n"
            "3. Get dimensions: N, C = x.shape[0], x.shape[1]\n"
            "4. Generate binary channel mask:\n"
            "   mask = np.random.binomial(1, 1-self.p, size=(N, C))\n"
            "5. Reshape mask for broadcasting:\n"
            "   mask_2d = mask.reshape((N, C, 1, 1))\n"
            "6. Apply mask and scale:\n"
            "   if self.inplace:\n"
            "       x *= mask_2d * self.scale\n"
            "       return x\n"
            "   else:\n"
            "       return x * mask_2d * self.scale\n"
            "7. Cache mask for backward pass"
        )

    def extra_repr(self) -> str:
        """Return extra representation with dropout probability info."""
        return f"p={self.p}"

    def train(self):
        """Enable training mode (activate dropout)."""
        self.training = True
        return self

    def eval(self):
        """Enable evaluation mode (deactivate dropout)."""
        self.training = False
        return self


class Dropout3d(Module):
    """
    Spatial Dropout for 3D Convolutional Networks (video/volumetric data).

    Drops entire feature maps across all spatial and temporal dimensions.

    Shapes:
    - Input: (N, C, D, H, W) for Conv3d output (video/3D medical imaging)
    - Output: (N, C, D, H, W)

    Parameters:
    -----------
    p: float
        Probability of dropping each channel

    Usage for video processing:
    ```
    class Conv3dBlock(Module):
        def __init__(self, in_channels, out_channels):
            self.conv = Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.dropout = Dropout3d(p=0.5)

        def forward(self, x):  # x: (N, C, T, H, W) temporal dimension
            x = self.conv(x)
            x = self.dropout(x)
            return x
    ```
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Initialize Dropout3d.

        Args:
            p: Dropout probability
            inplace: Whether to modify input in-place
        """
        super().__init__()

        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")

        self.p = p
        self.inplace = inplace
        self.scale = 1.0 / (1.0 - self.p) if self.p > 0 else 1.0

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply spatial dropout to 3D input.

        Args:
            x: Input tensor of shape (N, C, D, H, W)

        Returns:
            y: Output with dropped channels

        Algorithm:
        - Same as Dropout2d but for 5D input (N, C, D, H, W)
        - Mask shape: (N, C, 1, 1, 1)
        - Broadcasts across D, H, W
        """
        raise NotImplementedError(
            "Implement Dropout3d forward:\n"
            "1. Validate input shape is 5D: (N, C, D, H, W)\n"
            "2. If not self.training or self.p == 0: return x\n"
            "3. Generate channel mask (N, C)\n"
            "4. Reshape to (N, C, 1, 1, 1) for broadcasting\n"
            "5. Apply mask and scale\n"
            "6. Return output"
        )

    def extra_repr(self) -> str:
        """Return extra representation with dropout probability info."""
        return f"p={self.p}"

    def train(self):
        """Enable training mode."""
        self.training = True
        return self

    def eval(self):
        """Enable evaluation mode."""
        self.training = False
        return self


class DropPath(Module):
    """
    DropPath / Stochastic Depth - randomly skip residual blocks during training.

    Used in very deep networks (ResNets, Vision Transformers) to improve
    training stability and generalization by creating implicit ensembles
    of different depth networks.

    Shapes:
    - Input: any shape (typically residual block output)
    - Output: same shape as input

    Parameters:
    -----------
    p: float
        Probability of dropping the path (default: 0.0, no dropping)
        Typical values: 0.1-0.2 for deep networks
        0.0: No dropout (identity)
        0.5: Drop 50% of paths
        1.0: Always drop (not useful)

    inplace: bool
        Whether to modify input in-place (default: False)

    Properties:
    - Different behavior in training vs evaluation
    - Per-sample dropout (not per-element)
    - Particularly effective for residual networks
    - Works best with linear depth scheduling

    Usage Pattern in ResNet:
    ```
    class ResidualBlock(Module):
        def __init__(self, dim, drop_path=0.0):
            self.conv = Conv2d(dim, dim, 3, padding=1)
            self.drop_path = DropPath(drop_path)

        def forward(self, x):
            return x + self.drop_path(self.conv(x))
    ```

    Key Insight:
    When drop_path zeros the residual term, output becomes x (identity).
    This creates different effective depths during training.
    """

    def __init__(self, p: float = 0.0, inplace: bool = False):
        """
        Initialize DropPath.

        Args:
            p: Drop probability (0 to 1)
            inplace: Whether to modify input in-place
        """
        super().__init__()

        if not 0 <= p <= 1:
            raise ValueError(f"Drop probability must be in [0, 1], got {p}")

        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply stochastic depth to input.

        Args:
            x: Input tensor (typically residual block output)

        Returns:
            y: Output tensor (scaled by 1/(1-p) if not dropped, else zeros)

        Algorithm:
        1. If not training or p == 0: return x unchanged
        2. If training and p > 0:
           a. Generate single random value per sample: m ~ Bernoulli(1-p)
              Shape: (batch_size,) - same value for all elements of sample
           b. Reshape m to (batch_size, 1, 1, ...) to broadcast to x
           c. Scale: y = x * m / (1-p)
           d. If m=0 for sample: y=0 (path dropped, residual zeroed)
           e. If m=1 for sample: y=x/(1-p) (scaled to preserve expected value)
        3. Return output

        Key Implementation Detail:
        - Generate ONE mask value per sample (batch dimension)
        - Broadcast to all spatial/feature dimensions
        - NOT independent per element (like Dropout)
        - This is crucial for DropPath effectiveness

        Implementation Notes:
        - Mask shape: (batch,) not (batch, channels, height, width)
        - Reshape mask for broadcasting: (batch, 1, 1, ...)
        - Simpler than Dropout but more effective for residual blocks
        """
        raise NotImplementedError(
            "Implement DropPath forward:\n"
            "1. If not self.training or self.p == 0: return x\n"
            "2. Get batch size from x.shape[0]\n"
            "3. Generate per-sample mask (not per-element):\n"
            "   mask = np.random.binomial(1, 1-self.p, size=(batch_size,))\n"
            "4. Compute scale factor: scale = 1.0 / (1.0 - self.p)\n"
            "5. Reshape mask to broadcast:\n"
            "   - x has shape (batch, ...) \n"
            "   - reshape mask to (batch, 1, 1, ...)\n"
            "   - Keep batch dim, add 1s for other dims\n"
            "6. Apply mask and scale:\n"
            "   if self.inplace:\n"
            "       x *= mask * scale\n"
            "       return x\n"
            "   else:\n"
            "       return x * mask * scale\n"
            "7. Cache mask for backward pass"
        )

    def extra_repr(self) -> str:
        """Return extra representation with drop probability info."""
        return f"p={self.p}"

    def train(self):
        """Enable training mode."""
        self.training = True
        return self

    def eval(self):
        """Enable evaluation mode."""
        self.training = False
        return self


class DropPathScheduled(Module):
    """
    DropPath with depth-dependent scheduling.

    Linearly increases drop probability with depth - shallow layers rarely drop,
    deep layers drop more frequently.

    Empirically found to be crucial for stochastic depth effectiveness.

    Usage:
    ```
    drop_path_schedule = DropPathScheduled(p_base=0.2, total_depth=50)

    for layer_idx in range(50):
        drop_path = drop_path_schedule.get(layer_idx)
        block = ResidualBlock(dim, drop_path=drop_path)
    ```
    """

    def __init__(self, p_base: float = 0.2, total_depth: int = 50, schedule: str = "linear"):
        """
        Initialize scheduled DropPath.

        Args:
            p_base: Drop probability at deepest layer
            total_depth: Total number of layers in network
            schedule: Schedule type ("linear", "exponential")
        """
        super().__init__()

        if not 0 <= p_base <= 1:
            raise ValueError(f"p_base must be in [0, 1], got {p_base}")

        self.p_base = p_base
        self.total_depth = total_depth
        self.schedule = schedule

    def get(self, layer_idx: int) -> float:
        """
        Get drop probability for specific layer.

        Args:
            layer_idx: Layer index (0 to total_depth-1)

        Returns:
            p: Drop probability for this layer (0 at start, p_base at end)
        """
        raise NotImplementedError(
            "Implement depth scheduling:\n"
            "1. Validate layer_idx in [0, total_depth)\n"
            "2. Compute depth progress: depth_progress = layer_idx / total_depth\n"
            "3. Based on schedule:\n"
            "   - linear: p = p_base * depth_progress\n"
            "   - exponential: p = p_base * (depth_progress)^2\n"
            "4. Return p\n"
            "Note: First layer has p≈0, last layer has p≈p_base"
        )


class DropoutScheduled(Module):
    """
    Dropout with annealing schedule.

    Gradually decrease dropout probability during training.

    Motivation:
    - Early training: High dropout (strong regularization, learn robust features)
    - Late training: Low dropout (more neurons active, fine-tune)
    - Improves convergence and final performance

    Usage:
    ```
    dropout = DropoutScheduled(p_init=0.5, p_final=0.0, total_steps=1000)

    for step in range(1000):
        dropout.set_step(step)  # Update p based on step
        output = dropout.forward(x)
    ```
    """

    def __init__(
        self,
        p_init: float = 0.5,
        p_final: float = 0.0,
        total_steps: int = 1000,
        schedule: str = "linear",
    ):
        """
        Initialize scheduled dropout.

        Args:
            p_init: Initial dropout probability
            p_final: Final dropout probability
            total_steps: Total training steps
            schedule: Schedule type ("linear", "exponential", "cosine")
        """
        super().__init__()
        self.p_init = p_init
        self.p_final = p_final
        self.total_steps = total_steps
        self.schedule = schedule
        self.current_step = 0
        self.dropout = Dropout(p=p_init)

    def set_step(self, step: int):
        """Update dropout probability for current step."""
        raise NotImplementedError(
            "Implement schedule update:\n"
            "1. Compute progress: t = step / total_steps (0 to 1)\n"
            "2. Based on schedule type:\n"
            "   - linear: p = p_init + t * (p_final - p_init)\n"
            "   - exponential: p = p_init * (p_final/p_init)^t\n"
            "   - cosine: p = p_final + (p_init - p_final) * (1 + cos(π*t)) / 2\n"
            "3. Update self.dropout.p = new_p\n"
            "4. Update self.dropout.scale = 1/(1-new_p)"
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply scheduled dropout."""
        return self.dropout.forward(x)

    def train(self):
        """Enable training mode."""
        self.dropout.train()
        return self

    def eval(self):
        """Enable evaluation mode."""
        self.dropout.eval()
        return self


class LabelSmoothing(Module):
    """
    Label Smoothing Regularization.

    Replaces hard one-hot labels with smoothed soft labels to prevent
    model overconfidence and improve generalization.

    Parameters:
    -----------
    num_classes: int
        Number of classes

    smoothing: float
        Smoothing parameter ε ∈ [0, 1)
        ε=0: No smoothing (hard labels)
        ε=0.1: Standard smoothing (typical)
        ε=1: Complete uniformity (rarely used)

    Properties:
    - Works with any cross-entropy loss
    - Simple to implement (just modify labels)
    - No computational overhead
    - Improves model calibration
    - Particularly effective for large models

    Usage:
    ```
    smoother = LabelSmoothing(num_classes=1000, smoothing=0.1)

    # During training:
    hard_targets = Tensor([5, 12, 120, ...])  # class indices
    soft_targets = smoother.smooth_labels(hard_targets)  # smoothed labels

    # Compute loss with soft targets
    loss = cross_entropy(predictions, soft_targets)
    ```

    Mathematical Formula:
    y_smooth[i] = (1 - ε) * y_hard[i] + ε/k where k=num_classes

    For true class: y_smooth[true_class] = 1 - ε + ε/k = 1 - ε*(k-1)/k
    For other classes: y_smooth[other] = ε/k
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize Label Smoothing.

        Args:
            num_classes: Number of output classes
            smoothing: Smoothing parameter (typically 0.1)
        """
        super().__init__()

        if not 0 <= smoothing < 1:
            raise ValueError(f"Smoothing must be in [0, 1), got {smoothing}")

        self.num_classes = num_classes
        self.smoothing = smoothing

        # Precompute constants
        self.confidence = 1.0 - smoothing
        self.smooth_value = smoothing / (num_classes - 1)

    def smooth_labels(self, hard_labels: Tensor) -> Tensor:
        """
        Convert hard labels to smoothed soft labels.

        Args:
            hard_labels: 1D array of class indices, shape (batch_size,)
                        Values in range [0, num_classes)

        Returns:
            soft_labels: 2D array of smoothed labels, shape (batch_size, num_classes)
                        Values distributed according to label smoothing formula

        Algorithm:
        1. Create one-hot encoding of hard labels
        2. Replace 1s with (1-ε) and 0s with ε/(k-1)
        3. Equivalently: soft = (1-ε)*one_hot + ε/k

        Implementation Notes:
        - One-hot has shape (batch_size, num_classes)
        - Set all elements to ε/(k-1)
        - Set true class positions to (1-ε) + ε/(k-1)
        """
        raise NotImplementedError(
            "Implement label smoothing:\n"
            "1. Validate hard_labels shape: (batch_size,) and values in [0, num_classes)\n"
            "2. Create output array shape (batch_size, num_classes)\n"
            "3. Initialize all values to self.smooth_value = ε/(k-1)\n"
            "4. For each sample, set true class to self.confidence:\n"
            "   soft_labels[i, hard_labels[i]] = self.confidence\n"
            "5. Return soft_labels\n"
            "\n"
            "Alternatively (vectorized):\n"
            "1. Create one-hot encoded labels (batch_size, num_classes)\n"
            "2. soft = one_hot * self.confidence + (1 - one_hot) * self.smooth_value\n"
            "3. Return soft"
        )

    def extra_repr(self) -> str:
        """Return extra representation with smoothing parameter info."""
        return f"num_classes={self.num_classes}, smoothing={self.smoothing}"

    def __call__(self, hard_labels: Tensor) -> Tensor:
        """Convenience method to apply label smoothing."""
        return self.smooth_labels(hard_labels)


class LabelSmoothingCrossEntropy(Module):
    """
    Cross-Entropy Loss with Label Smoothing integrated.

    Combines label smoothing and cross-entropy computation for numerical stability
    and convenience.

    This is often more efficient than separately smoothing labels then computing loss.

    Parameters:
    -----------
    num_classes: int
        Number of classes

    smoothing: float
        Smoothing parameter

    Usage:
    ```
    criterion = LabelSmoothingCrossEntropy(num_classes=1000, smoothing=0.1)

    # In training loop:
    predictions = model(batch)  # shape (batch, num_classes)
    targets = batch_labels      # shape (batch,) with class indices
    loss = criterion(predictions, targets)
    ```

    Mathematical Implementation:
    L = -Σ_c y_smooth[c] * log(p[c])
      = -(1-ε)*log(p_true) - Σ_{c≠true} (ε/(k-1))*log(p[c])

    This formulation is numerically stable and efficient.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize Cross-Entropy with Label Smoothing.

        Args:
            num_classes: Number of classes
            smoothing: Smoothing parameter
        """
        super().__init__()

        self.num_classes = num_classes
        self.smoothing = smoothing
        self.label_smoother = LabelSmoothing(num_classes, smoothing)

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> float:
        """
        Compute cross-entropy loss with label smoothing.

        Args:
            predictions: Model predictions, shape (batch_size, num_classes)
                        Can be logits or probabilities
            targets: Target class indices, shape (batch_size,)
                    Values in range [0, num_classes)

        Returns:
            loss: Scalar loss value

        Algorithm:
        1. Smooth the target labels
        2. Compute cross-entropy with smoothed labels
        3. Return scalar loss

        Implementation Notes:
        - If predictions are logits, apply softmax first
        - Use log-softmax for numerical stability: log(p[c]) = logits[c] - logsumexp(logits)
        - Avoid exp() of large numbers (numerical overflow)

        Efficient implementation:
        1. Compute log softmax: log_p = logits - logsumexp(logits, axis=1)
        2. Smooth targets
        3. Loss = -sum(y_smooth * log_p)
        """
        raise NotImplementedError(
            "Implement cross-entropy with label smoothing:\n"
            "1. Compute softmax probabilities from predictions\n"
            "   For numerical stability, use log-softmax:\n"
            "   log_p = predictions - np.logaddexp.reduce(predictions, axis=1, keepdims=True)\n"
            "   p = np.exp(log_p)  # if needed\n"
            "2. Smooth the target labels:\n"
            "   soft_targets = self.label_smoother.smooth_labels(targets)\n"
            "   Shape: (batch_size, num_classes)\n"
            "3. Compute cross-entropy:\n"
            "   loss_per_sample = -np.sum(soft_targets * log_p, axis=1)\n"
            "4. Return mean or sum of per-sample losses:\n"
            "   return np.mean(loss_per_sample)\n"
            "5. Store for backward pass if implementing gradient"
        )

    def extra_repr(self) -> str:
        """Return extra representation with loss parameters info."""
        return f"num_classes={self.num_classes}, smoothing={self.smoothing}"
