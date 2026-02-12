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
from .regularization_functional import dropout, dropout1d, dropout2d


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
        return dropout(x, p=self.p, training=self.training)

    def extra_repr(self) -> str:
        """Return extra representation with dropout probability info."""
        return f"p={self.p}"


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
        return dropout1d(x, p=self.p, training=self.training)

    def extra_repr(self) -> str:
        """Return extra representation with dropout probability info."""
        return f"p={self.p}"


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
        return dropout2d(x, p=self.p, training=self.training)

    def extra_repr(self) -> str:
        """Return extra representation with dropout probability info."""
        return f"p={self.p}"


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
        return dropout3d(x, p=self.p, training=self.training)

    def extra_repr(self) -> str:
        """Return extra representation with dropout probability info."""
        return f"p={self.p}"


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

