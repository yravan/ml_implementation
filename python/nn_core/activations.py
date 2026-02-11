"""
Activation Functions
====================

Neural network activation modules and functional interfaces.

This module provides activation layers that can be used in neural network architectures.
All modules take Tensor inputs and return Tensor outputs, with gradients computed
automatically via the computational graph (no backward() methods needed).

Modules:
- ReLU family: ReLU, LeakyReLU, PReLU, ELU, SELU, ReLU6
- GELU family: GELU, QuickGELU
- Sigmoid family: Sigmoid, LogSigmoid, HardSigmoid
- Softmax family: Softmax, LogSoftmax, Softmax2D
- Tanh family: Tanh, Hardtanh, Tanhshrink

Functional interfaces are also provided for stateless usage.

References
----------
- "Rectified Linear Units Improve Restricted Boltzmann Machines" (ReLU)
  https://icml.cc/Conferences/2010/papers/432.pdf
- "Gaussian Error Linear Units (GELUs)" Hendrycks & Gimpel (2016)
  https://arxiv.org/abs/1606.08415
- "Self-Normalizing Neural Networks" (SELU) (2017)
  https://arxiv.org/abs/1706.02515
"""

# Implementation Status: STUBS
# Complexity: Easy to Medium
# Prerequisites: foundations/computational_graph, foundations/functionals

import numpy as np
from typing import Optional, Union, Tuple, Literal

from .activations_functional import (
    leaky_relu,
    prelu,
    elu,
    selu,
    relu6,
    gelu,
    quickgelu,
    silu,
    softplus,
    softsign,
    mish,
    tanhshrink,
    hardtanh,
    tanh,
    hard_sigmoid,
    relu,
)
from .module import Module, Parameter
from python.foundations import Tensor, maximum
from ..foundations.computational_graph import logsoftmax, softmax, logsigmoid, sigmoid


# =============================================================================
# ReLU Family
# =============================================================================

class ReLU(Module):
    """
    Rectified Linear Unit: f(x) = max(0, x)

    The most commonly used activation function in deep learning.

    Properties:
    - Non-saturating (no vanishing gradient for positive values)
    - Computationally efficient (just a threshold)
    - Induces sparsity (many zeros)

    Example:
        >>> relu = ReLU()
        >>> x = Tensor(np.array([-2, -1, 0, 1, 2]))
        >>> relu(x)
        Tensor([0, 0, 0, 1, 2])
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: f(x) = max(0, x)

        Args:
            x: Input Tensor of any shape

        Returns:
            ReLU(x), same shape as input
        """
        return relu(x)


class LeakyReLU(Module):
    """
    Leaky ReLU: f(x) = x if x > 0, else αx

    Allows a small gradient when the unit is not active, preventing
    the "dying ReLU" problem.

    Example:
        >>> lrelu = LeakyReLU(negative_slope=0.1)
        >>> x = Tensor(np.array([-2, -1, 0, 1, 2]))
        >>> lrelu(x)
        Tensor([-0.2, -0.1, 0., 1., 2.])
    """

    def __init__(self, negative_slope: float = 0.01):
        """
        Initialize LeakyReLU.

        Args:
            negative_slope: Slope for negative values (default: 0.01)
        """
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return leaky_relu(x, alpha=self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class PReLU(Module):
    """
    Parametric ReLU: f(x) = x if x > 0, else α*x

    Like LeakyReLU but α is a learned parameter.

    Attributes:
        weight: Learnable negative slope, shape (num_parameters,)
    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        """
        Initialize PReLU.

        Args:
            num_parameters: Number of α parameters (1 or equal to channels)
            init: Initial value for α
        """
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = Parameter(np.full(num_parameters, init))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with learnable negative slope."""
        return prelu(x, self.weight)

    def extra_repr(self) -> str:
        return f"num_parameters={self.num_parameters}"


class ELU(Module):
    """
    Exponential Linear Unit: f(x) = x if x > 0, else α(e^x - 1)

    Properties:
    - Smooth (continuous derivative)
    - Pushes mean activations closer to 0
    - Negative values provide noise robustness

    Example:
        >>> elu = ELU(alpha=1.0)
        >>> x = Tensor(np.array([-2, -1, 0, 1, 2]))
        >>> elu(x)
        Tensor([-0.8647, -0.6321, 0., 1., 2.])
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize ELU.

        Args:
            alpha: Scale for negative values
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return elu(x, self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class SELU(Module):
    """
    Scaled Exponential Linear Unit: f(x) = λ * (x if x > 0 else α(e^x - 1))

    Self-normalizing: with proper weight initialization, activations
    automatically converge to zero mean and unit variance.

    IMPORTANT: Use LeCun normal initialization for SELU to work properly:
    W ~ N(0, sqrt(1/fan_in))

    Fixed parameters from the paper:
    α ≈ 1.6732632423543772848170429916717
    λ ≈ 1.0507009873554804934193349852946
    """

    def __init__(self):
        """Initialize SELU (no learnable parameters)."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return selu(x)

    def extra_repr(self) -> str:
        return f"alpha={self.ALPHA:.4f}, scale={self.SCALE:.4f}"


class ReLU6(Module):
    """
    ReLU6: f(x) = min(max(0, x), 6)

    Used in MobileNet and other mobile architectures.
    The upper bound helps with fixed-point quantization.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: clip to [0, 6]."""
        return relu6(x)


# =============================================================================
# GELU Family
# =============================================================================

class GELU(Module):
    """
    Gaussian Error Linear Unit.

    GELU(x) = x * Φ(x) where Φ is the Gaussian CDF.

    Standard activation for Transformer models (BERT, GPT, ViT).

    Example:
        >>> gelu = GELU()
        >>> x = Tensor(np.array([-2, -1, 0, 1, 2]))
        >>> gelu(x)
        Tensor([-0.0455, -0.1588, 0., 0.8413, 1.9545])
    """

    SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
    COEF = 0.044715

    def __init__(self, approximate: bool = True):
        """
        Initialize GELU.

        Args:
            approximate: If True, use fast tanh approximation (recommended).
                        If False, use exact formula with erf.
        """
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Approximate (default):
            GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

        Exact:
            GELU(x) = x * Φ(x) = 0.5 * x * (1 + erf(x / √2))
        """
        return gelu(x, self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate}"


class QuickGELU(Module):
    """
    Quick GELU approximation using sigmoid.

    QuickGELU(x) = x * sigmoid(1.702 * x)

    Even faster than tanh approximation, used in some CLIP models.
    """

    SCALE = 1.702

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return quickgelu(x, self.SCALE)


class SiLU(Module):
    """
    Sigmoid Linear Unit (also known as Swish).

    SiLU(x) = x * sigmoid(x)

    Discovered via neural architecture search. Related to GELU but
    uses sigmoid instead of Gaussian CDF.

    Reference: "Searching for Activation Functions" (2017)
    https://arxiv.org/abs/1710.05941
    """
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return silu(x)


# =============================================================================
# Sigmoid Family
# =============================================================================

class Sigmoid(Module):
    """
    Sigmoid activation: σ(x) = 1 / (1 + e^(-x))

    Squashes input to (0, 1). Used for binary classification outputs
    and gating mechanisms.

    Example:
        >>> sigmoid = Sigmoid()
        >>> x = Tensor(np.array([-2, -1, 0, 1, 2]))
        >>> sigmoid(x)
        Tensor([0.119, 0.269, 0.5, 0.731, 0.881])
    """

    def __init__(self):
        """Initialize Sigmoid."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: σ(x) = 1 / (1 + e^(-x))

        Args:
            x: Input Tensor of any shape

        Returns:
            Sigmoid of x, same shape, values in (0, 1)
        """
        return sigmoid(x)


class LogSigmoid(Module):
    """
    Log Sigmoid: f(x) = log(σ(x)) = log(1 / (1 + e^(-x)))

    More numerically stable than computing log(sigmoid(x)) separately.
    Useful for binary cross-entropy loss.

    f(x) = -log(1 + e^(-x)) = -softplus(-x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return logsigmoid(x)


class HardSigmoid(Module):
    """
    Hard Sigmoid: piecewise linear approximation to sigmoid.

    f(x) = clip(slope*x + offset, 0, 1)

    Default: f(x) = clip(0.2*x + 0.5, 0, 1)

    Faster than sigmoid (no exp), used in efficient architectures.
    """

    def __init__(self, slope: float = 0.2, offset: float = 0.5):
        """
        Initialize HardSigmoid.

        Args:
            slope: Slope of linear region
            offset: Offset at x=0
        """
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return hard_sigmoid(x, self.slope, self.offset)

    def extra_repr(self) -> str:
        return f"slope={self.slope}, offset={self.offset}"


# =============================================================================
# Softmax Family
# =============================================================================

class Softmax(Module):
    """
    Softmax activation: converts logits to probabilities.

    Softmax(x)_i = exp(x_i) / Σ_j exp(x_j)

    Example:
        >>> softmax = Softmax(axis=-1)
        >>> logits = Tensor(np.array([[1.0, 2.0, 3.0],
        ...                           [1.0, 1.0, 1.0]]))
        >>> probs = softmax(logits)
        >>> probs.data.sum(axis=-1)  # Sums to 1 along last axis
        array([1., 1.])
    """

    def __init__(self, axis: Union[int, Tuple[int, ...]] = -1):
        """
        Initialize Softmax.

        Args:
            axis: Axis along which to compute softmax
        """
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: numerically stable softmax.

        Args:
            x: Input logits, shape (*, num_classes) where * is batch dims

        Returns:
            Probabilities, same shape, summing to 1 along self.axis
        """
        return softmax(x, axis=self.axis)

    def extra_repr(self) -> str:
        return f"axis={self.axis}"


class LogSoftmax(Module):
    """
    Log Softmax: log(softmax(x))

    More numerically stable than computing log(softmax(x)) separately.
    Combined with NLLLoss, this gives cross-entropy loss.

    log_softmax(x)_i = x_i - logsumexp(x)
    """

    def __init__(self, axis: Union[int, Tuple[int, ...]] = -1):
        """Initialize LogSoftmax."""
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return logsoftmax(x, axis=self.axis)
    def extra_repr(self) -> str:
        return f"axis={self.axis}"


class Softmax2D(Module):
    """
    Softmax over spatial dimensions for segmentation.

    For input (batch, channels, height, width), applies softmax over channels
    independently at each spatial location.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Apply softmax over channel dimension (axis=1)."""
        return softmax(x, axis=1)


# =============================================================================
# Tanh Family
# =============================================================================

class Tanh(Module):
    """
    Hyperbolic tangent activation: tanh(x)

    Squashes input to (-1, 1). Zero-centered alternative to sigmoid.

    Properties:
    - Zero-centered: outputs can be negative
    - Bounded: prevents exploding activations
    - Saturating: gradients vanish for large |x|

    Example:
        >>> tanh = Tanh()
        >>> x = Tensor(np.array([-2, -1, 0, 1, 2]))
        >>> tanh(x)
        Tensor([-0.964, -0.762, 0., 0.762, 0.964])
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: tanh(x)

        Args:
            x: Input Tensor of any shape

        Returns:
            tanh(x), same shape, values in (-1, 1)
        """
        return tanh(x)


class Hardtanh(Module):
    """
    Hard Tanh: piecewise linear approximation.

    f(x) = min_val if x < min_val
           x       if min_val <= x <= max_val
           max_val if x > max_val

    Faster to compute than tanh, used in efficient architectures.
    Also known as "clamp" activation.
    """

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        """
        Initialize Hardtanh.

        Args:
            min_val: Minimum output value
            max_val: Maximum output value
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: clip to [min_val, max_val]."""
        return hardtanh(x, self.min_val, self.max_val)

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}, max_val={self.max_val}"


class Tanhshrink(Module):
    """
    Tanhshrink: f(x) = x - tanh(x)

    Shrinks values toward zero, with larger shrinkage for larger |x|.
    Used in some sparse coding and denoising applications.
    """
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return tanhshrink(x)


# =============================================================================
# Other Activations
# =============================================================================

class Softplus(Module):
    """
    Softplus: f(x) = (1/β) * log(1 + exp(β*x))

    A smooth approximation to ReLU. As β → ∞, approaches ReLU.

    Properties:
    - Always positive
    - Smooth everywhere (unlike ReLU at 0)
    - Derivative is sigmoid: f'(x) = sigmoid(β*x)
    """

    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        """
        Initialize Softplus.

        Args:
            beta: Scaling factor
            threshold: Above this, use linear approximation for stability
        """
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return softplus(x, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, threshold={self.threshold}"


class Softsign(Module):
    """
    Softsign: f(x) = x / (1 + |x|)

    Similar to tanh but with polynomial decay instead of exponential.
    Converges more slowly to ±1.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return softsign(x)


class Mish(Module):
    """
    Mish: f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

    A self-regularized non-monotonic activation function.

    Reference: "Mish: A Self Regularized Non-Monotonic Activation Function" (2019)
    https://arxiv.org/abs/1908.08681
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return mish(x)


class Threshold(Module):
    """
    Threshold: f(x) = x if x > threshold, else value

    Simple thresholding activation.
    """

    def __init__(self, threshold: float, value: float):
        """
        Initialize Threshold.

        Args:
            threshold: Threshold value
            value: Value to use when x <= threshold
        """
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        mask = x > self.threshold
        out = x.copy()
        out.set(mask, self.value)
        return out

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, value={self.value}"

