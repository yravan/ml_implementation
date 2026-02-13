"""
Regularization Functional Operations
=====================================

This module provides functional operations for regularization layers.
Function classes handle the forward/backward computation with np.ndarray,
while Module classes in regularization.py wrap these for Tensor operations.

Function Classes:
    - Dropout: Standard element-wise dropout
    - Dropout1d: Spatial dropout for 1D (channels dropped)
    - Dropout2d: Spatial dropout for 2D (channels dropped)
    - Dropout3d: Spatial dropout for 3D (channels dropped)
    - DropPath: Stochastic depth for residual networks
    - DropBlock: Block-wise dropout for CNNs
    - AlphaDropout: Dropout for self-normalizing networks

Helper Functions:
    - dropout, dropout1d, dropout2d, dropout3d
    - drop_path, drop_block, alpha_dropout
"""

import numpy as np
from typing import Tuple, Optional

from python.foundations import Function, convert_to_function, _no_grad


# =============================================================================
# Standard Dropout Function Classes
# =============================================================================

class Dropout(Function):
    """
    Standard Dropout functional operation.

    Randomly sets elements to zero during training with probability p.
    Scales remaining elements by 1/(1-p) to maintain expected values.

    Math:
        Training:
            mask ~ Bernoulli(1 - p)
            y = x * mask / (1 - p)

        Inference:
            y = x

    Note: Uses "inverted dropout" where scaling happens at training time.
    """

    def forward(
        self,
        x: np.ndarray,
        p: float = 0.5,
        training: bool = True
    ) -> np.ndarray:
        """
        Apply dropout.

        Args:
            x: Input array
            p: Probability of dropping an element (0 to 1)
            training: Whether in training mode

        Returns:
            Output array with dropout applied (if training)
        """
        if not training:
            return x
        mask = (np.random.rand(*x.shape) >= p).astype(x.dtype)
        global _no_grad
        if not _no_grad:
            self.mask = mask
            self.p = p
        return x * mask / (1 - p)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient for dropout.

        Args:
            grad_output: Gradient w.r.t. output

        Returns:
            Tuple of (grad_input,)
        """
        dx = grad_output / (1 - self.p) * self.mask
        return dx,


class Dropout1d(Function):
    """
    1D Spatial Dropout functional operation.

    Drops entire channels (feature maps) for 1D inputs.
    Same mask across all temporal/spatial positions.

    Input shape: (batch, channels, length)
    Mask shape: (batch, channels, 1) - broadcasts across length
    """

    def forward(
        self,
        x: np.ndarray,
        p: float = 0.5,
        training: bool = True
    ) -> np.ndarray:
        """
        Apply 1D spatial dropout.

        Args:
            x: Input array (batch, channels, length)
            p: Probability of dropping a channel
            training: Whether in training mode

        Returns:
            Output array
        """
        if not training:
            return x
        mask = (np.random.rand(*x.shape[:2]) >= p).astype(x.dtype)[..., None]
        global _no_grad
        if not _no_grad:
            self.mask = mask
            self.p = p
        return x * mask / (1 - p)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Compute gradient for 1D spatial dropout."""
        dx = grad_output / (1 - self.p) * self.mask
        return dx,


class Dropout2d(Function):
    """
    2D Spatial Dropout functional operation.

    Drops entire channels (feature maps) for 2D inputs.
    Same mask across all spatial positions.

    Input shape: (batch, channels, height, width)
    Mask shape: (batch, channels, 1, 1) - broadcasts across spatial dims
    """

    def forward(
        self,
        x: np.ndarray,
        p: float = 0.5,
        training: bool = True
    ) -> np.ndarray:
        """
        Apply 2D spatial dropout.

        Args:
            x: Input array (batch, channels, height, width)
            p: Probability of dropping a channel
            training: Whether in training mode

        Returns:
            Output array
        """
        if not training:
            return x
        mask = (np.random.rand(*x.shape[:2]) >= p).astype(x.dtype)[..., None, None]
        global _no_grad
        if not _no_grad:
            self.mask = mask
            self.p = p
        return x * mask / (1 - p)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Compute gradient for 1D spatial dropout."""
        dx = grad_output / (1 - self.p) * self.mask
        return (dx,)


class Dropout3d(Function):
    """
    3D Spatial Dropout functional operation.

    Drops entire channels for 3D inputs (video/volumetric data).
    Same mask across all spatial and temporal positions.

    Input shape: (batch, channels, depth, height, width)
    Mask shape: (batch, channels, 1, 1, 1)
    """

    def forward(
        self,
        x: np.ndarray,
        p: float = 0.5,
        training: bool = True
    ) -> np.ndarray:
        """
        Apply 3D spatial dropout.

        Args:
            x: Input array (batch, channels, depth, height, width)
            p: Probability of dropping a channel
            training: Whether in training mode

        Returns:
            Output array
        """
        if not training:
            return x
        mask = (np.random.rand(*x.shape[:2]) >= p).astype(x.dtype)[..., None, None, None]
        global _no_grad
        if not _no_grad:
            self.mask = mask
            self.p = p
        return x * mask / (1 - p)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Compute gradient for 1D spatial dropout."""
        dx = grad_output / (1 - self.p) * self.mask
        return dx,


# =============================================================================
# DropPath (Stochastic Depth) Function Class
# =============================================================================

class DropPath(Function):
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

    References:
        - Deep Networks with Stochastic Depth (Huang et al., 2016)
    """

    def forward(
        self,
        x: np.ndarray,
        drop_prob: float = 0.0,
        training: bool = True
    ) -> np.ndarray:
        """
        Apply drop path.

        Args:
            x: Input array (batch, ...)
            drop_prob: Probability of dropping a sample's path
            training: Whether in training mode

        Returns:
            Output array
        """
        raise NotImplementedError(
            "TODO: Implement DropPath forward\n"
            "Hint:\n"
            "  global _no_grad\n"
            "  \n"
            "  if not training or drop_prob == 0:\n"
            "      return x\n"
            "  \n"
            "  keep_prob = 1 - drop_prob\n"
            "  \n"
            "  # Mask shape: (batch, 1, 1, ...) - one value per sample\n"
            "  shape = (x.shape[0],) + (1,) * (x.ndim - 1)\n"
            "  mask = (np.random.rand(*shape) < keep_prob).astype(x.dtype)\n"
            "  \n"
            "  if not _no_grad:\n"
            "      self.mask = mask\n"
            "      self.keep_prob = keep_prob\n"
            "      self.training = training\n"
            "  \n"
            "  return x * mask / keep_prob"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient for drop path.

        Args:
            grad_output: Gradient w.r.t. output

        Returns:
            Tuple of (grad_input,)
        """
        raise NotImplementedError(
            "TODO: Implement DropPath backward\n"
            "Hint:\n"
            "  if not self.training or self.keep_prob == 1:\n"
            "      return (grad_output,)\n"
            "  \n"
            "  return (grad_output * self.mask / self.keep_prob,)"
        )


# =============================================================================
# DropBlock Function Class
# =============================================================================

class DropBlock(Function):
    """
    DropBlock regularization functional operation.

    Drops contiguous regions (blocks) of feature maps.
    More effective than standard dropout for convolutional networks.

    Input shape: (batch, channels, height, width)

    Math:
        1. Sample mask centers with probability gamma
        2. Expand each center to a block_size x block_size region
        3. Apply mask and rescale

    References:
        - DropBlock (Ghiasi et al., 2018): https://arxiv.org/abs/1810.12890
    """

    def forward(
        self,
        x: np.ndarray,
        drop_prob: float = 0.1,
        block_size: int = 7,
        training: bool = True
    ) -> np.ndarray:
        """
        Apply DropBlock.

        Args:
            x: Input array (batch, channels, height, width)
            drop_prob: Target probability of dropping a unit
            block_size: Size of block to drop
            training: Whether in training mode

        Returns:
            Output array
        """
        raise NotImplementedError(
            "TODO: Implement DropBlock forward\n"
            "Hint:\n"
            "  global _no_grad\n"
            "  \n"
            "  if not training or drop_prob == 0:\n"
            "      return x\n"
            "  \n"
            "  batch, channels, h, w = x.shape\n"
            "  \n"
            "  # Compute gamma (probability of sampling block center)\n"
            "  valid_h = h - block_size + 1\n"
            "  valid_w = w - block_size + 1\n"
            "  gamma = (drop_prob * h * w) / (block_size ** 2 * valid_h * valid_w)\n"
            "  \n"
            "  # Sample block centers\n"
            "  centers = np.random.rand(batch, channels, valid_h, valid_w) < gamma\n"
            "  \n"
            "  # Expand centers to blocks\n"
            "  mask = np.ones((batch, channels, h, w), dtype=x.dtype)\n"
            "  for i in range(block_size):\n"
            "      for j in range(block_size):\n"
            "          mask[:, :, i:i+valid_h, j:j+valid_w] *= ~centers\n"
            "  \n"
            "  # Normalize by fraction of kept elements\n"
            "  count = mask.sum()\n"
            "  if count > 0:\n"
            "      return x * mask * (mask.size / count)\n"
            "  return x * mask"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Compute gradient for DropBlock."""
        raise NotImplementedError("TODO: Implement DropBlock backward")


# =============================================================================
# AlphaDropout Function Class
# =============================================================================

class AlphaDropout(Function):
    """
    Alpha Dropout for Self-Normalizing Neural Networks (SNNs).

    Unlike standard dropout, Alpha Dropout maintains mean and variance
    of activations, making it suitable for networks using SELU activation.

    Math:
        Dropped values are set to saturation point α' (not zero)
        Remaining values are scaled and shifted to maintain statistics:
        y = a * (x * mask + α' * (1 - mask)) + b

    References:
        - Self-Normalizing Neural Networks (Klambauer et al., 2017)
    """

    # SELU parameters
    ALPHA = 1.6732632423543772848170429916717
    SCALE = 1.0507009873554804934193349852946

    def forward(
        self,
        x: np.ndarray,
        p: float = 0.5,
        training: bool = True
    ) -> np.ndarray:
        """
        Apply alpha dropout.

        Args:
            x: Input array
            p: Probability of dropping an element
            training: Whether in training mode

        Returns:
            Output array
        """
        raise NotImplementedError(
            "TODO: Implement AlphaDropout forward\n"
            "Hint:\n"
            "  global _no_grad\n"
            "  \n"
            "  if not training or p == 0:\n"
            "      return x\n"
            "  \n"
            "  # Saturation point for negative inputs under SELU\n"
            "  alpha_prime = -self.SCALE * self.ALPHA\n"
            "  \n"
            "  # Keep probability\n"
            "  q = 1 - p\n"
            "  \n"
            "  # Compute a and b to maintain mean/variance\n"
            "  a = (q + alpha_prime ** 2 * p * q) ** (-0.5)\n"
            "  b = -a * (p * alpha_prime)\n"
            "  \n"
            "  # Generate mask\n"
            "  mask = (np.random.rand(*x.shape) > p).astype(x.dtype)\n"
            "  \n"
            "  if not _no_grad:\n"
            "      self.mask = mask\n"
            "      self.a = a\n"
            "      self.p = p\n"
            "      self.training = training\n"
            "  \n"
            "  # Apply: kept elements stay, dropped become alpha'\n"
            "  y = x * mask + alpha_prime * (1 - mask)\n"
            "  \n"
            "  # Scale and shift\n"
            "  return a * y + b"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient for alpha dropout.

        Args:
            grad_output: Gradient w.r.t. output

        Returns:
            Tuple of (grad_input,)
        """
        raise NotImplementedError(
            "TODO: Implement AlphaDropout backward\n"
            "Hint:\n"
            "  if not self.training or self.p == 0:\n"
            "      return (grad_output,)\n"
            "  \n"
            "  # Gradient only flows through kept elements\n"
            "  return (grad_output * self.a * self.mask,)"
        )

