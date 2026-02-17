"""
Weight Initialization Functions
===============================

This module provides weight initialization methods for neural networks.
Proper initialization is crucial for effective training of deep networks.

All functions follow the PyTorch convention of in-place initialization with
trailing underscore (`_`) and take a Tensor as the first argument.

Available Functions:
    - xavier_uniform_, xavier_normal_: Xavier/Glorot initialization
    - kaiming_uniform_, kaiming_normal_: Kaiming/He initialization
    - normal_, uniform_: Basic random initialization
    - zeros_, ones_, constant_: Constant initialization
    - orthogonal_: Orthogonal matrix initialization

Helper Functions:
    - calculate_fan_in_fan_out: Compute fan-in and fan-out for weight tensors

Configuration Classes:
    - GainConfig: Recommended gain values for activations
    - ActivationConfig: Activation slope parameters

Theory Notes:
=============

XAVIER (GLOROT) INITIALIZATION:
- Designed for sigmoid/tanh activations
- Maintains signal variance through layers
- Formula: Var[w] = 2 / (fan_in + fan_out)

KAIMING (HE) INITIALIZATION:
- Designed for ReLU and variants
- Compensates for ReLU zeroing out half the signal
- Formula: Var[w] = 2 / fan_in

ORTHOGONAL INITIALIZATION:
- Preserves vector norms: ||Wx|| = ||x||
- Excellent for deep networks and RNNs
- Uses QR decomposition to generate orthogonal matrices

REFERENCES:
- "Understanding the difficulty of training deep feedforward neural networks"
  Glorot & Bengio, 2010 - Xavier initialization
- "Delving Deep into Rectifiers" He et al., 2015 - Kaiming initialization
- "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
  Saxe et al., 2013 - Orthogonal initialization
"""

import math
import numpy as np
from typing import Literal, Tuple, Union

from ..foundations import Tensor


def _as_np(tensor):
    """Return the underlying numpy array for a Tensor or a numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.data  # Tensor.data is a numpy array


def _set_data(tensor, values):
    """Write *values* into tensor, works for both Tensor and numpy array."""
    if isinstance(tensor, np.ndarray):
        tensor[:] = values
    else:
        tensor.data = values


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_fan_in_fan_out(tensor: Tensor) -> Tuple[int, int]:
    """
    Calculate fan-in and fan-out for a tensor.

    For different tensor shapes:
    - (out_features, in_features): Linear layers
    - (out_channels, in_channels, kernel_h, kernel_w): Convolution layers
    - (features,): Bias/1D tensors -> fan_in = fan_out = features

    Args:
        tensor: Weight tensor

    Returns:
        Tuple of (fan_in, fan_out)

    Raises:
        ValueError: If tensor has 0 dimensions

    Example:
        >>> w_linear = Tensor(np.empty((10, 20)))
        >>> fan_in, fan_out = calculate_fan_in_fan_out(w_linear)
        >>> assert fan_in == 20 and fan_out == 10

        >>> w_conv = Tensor(np.empty((64, 3, 3, 3)))  # out, in, h, w
        >>> fan_in, fan_out = calculate_fan_in_fan_out(w_conv)
        >>> assert fan_in == 3 * 3 * 3 and fan_out == 64 * 3 * 3
    """
    if tensor.ndim == 0:
        raise ValueError("tensor cannot have dimension 0")
    if tensor.ndim == 1:
        return tensor.shape[0], tensor.shape[0]
    if tensor.ndim == 2:
        return tensor.shape[1], tensor.shape[0]
    receptive_field = math.prod(tensor.shape[2:])
    fan_in = tensor.shape[1] * receptive_field
    fan_out = tensor.shape[0] * receptive_field
    return fan_in, fan_out


# =============================================================================
# Configuration Classes
# =============================================================================

class GainConfig:
    """
    Recommended gain values for different activation functions.

    Used with Xavier initialization to adjust variance scaling
    for specific activation functions.

    Reference: PyTorch nn.init documentation
    """

    LINEAR = 1.0
    SIGMOID = 1.0
    TANH = 5.0 / 3.0
    RELU = math.sqrt(2.0)
    LEAKY_RELU_01 = math.sqrt(2.0 / (1 + 0.1**2))
    SELU = 3.0 / 4.0

    @staticmethod
    def for_activation(activation_name: str, param: float = 0.01) -> float:
        """
        Get recommended gain for activation function.

        Args:
            activation_name: Name of activation ('sigmoid', 'tanh', 'relu',
                           'leaky_relu', 'selu')
            param: Negative slope for leaky_relu (default 0.01)

        Returns:
            Recommended gain value

        Raises:
            ValueError: If activation not known
        """
        raise NotImplementedError(
            "TODO: Map activation names to gain values\n"
            "1. Convert activation_name to lowercase\n"
            "2. Return appropriate gain:\n"
            "   - 'linear', 'identity' -> 1.0\n"
            "   - 'sigmoid' -> 1.0\n"
            "   - 'tanh' -> 5.0 / 3.0\n"
            "   - 'relu' -> sqrt(2.0)\n"
            "   - 'leaky_relu' -> sqrt(2.0 / (1 + param**2))\n"
            "   - 'selu' -> 3.0 / 4.0\n"
            "3. Raise ValueError for unknown activation"
        )


class ActivationConfig:
    """
    Configuration for different activation functions.

    Maps activation names to their 'a' (negative slope) parameters
    for use with Kaiming initialization.
    """

    RELU = 0.0
    LEAKY_RELU = 0.01
    SELU = 0.0

    @staticmethod
    def get_a(nonlinearity: str) -> float:
        """
        Get 'a' parameter for activation function.

        Args:
            nonlinearity: Activation name ('relu', 'leaky_relu', 'selu')

        Returns:
            Negative slope parameter 'a'

        Raises:
            ValueError: If activation not known
        """
        raise NotImplementedError(
            "TODO: Map activation names to 'a' values\n"
            "1. Convert nonlinearity to lowercase\n"
            "2. Return appropriate 'a':\n"
            "   - 'relu' -> 0.0\n"
            "   - 'leaky_relu' -> 0.01\n"
            "   - 'selu' -> 0.0\n"
            "3. Raise ValueError for unknown nonlinearity"
        )


# =============================================================================
# Xavier/Glorot Initialization
# =============================================================================

def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> None:
    """
    Initialize tensor with Xavier uniform distribution (in-place).

    Initializes tensor with values drawn from U[-limit, limit] where:
    limit = gain * sqrt(6 / (fan_in + fan_out))

    Args:
        tensor: Tensor to initialize (modified in-place)
        gain: Multiplicative constant for limit adjustment
             (default 1.0, use 5/3 for tanh, sqrt(2) for ReLU)

    Returns:
        The initialized tensor (same as input)

    Raises:
        ValueError: If tensor has less than 1 dimension

    Example:
        >>> weight = Tensor(np.empty((10, 20)))
        >>> xavier_uniform_(weight)
        >>> # weight now contains values from U[-limit, limit]

        >>> # With gain for different activations
        >>> xavier_uniform_(weight, gain=5.0/3.0)  # For tanh
        >>> xavier_uniform_(weight, gain=math.sqrt(2))  # For ReLU
    """
    fan_in, fan_out = calculate_fan_in_fan_out(tensor)
    limit = gain * np.sqrt(6 / (fan_in + fan_out))
    arr = _as_np(tensor)
    _set_data(tensor, np.random.uniform(-limit, limit, arr.shape).astype(arr.dtype, copy=False))


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> None:
    """
    Initialize tensor with Xavier normal distribution (in-place).

    Initializes tensor with values drawn from N(0, std) where:
    std = gain * sqrt(2 / (fan_in + fan_out))

    Args:
        tensor: Tensor to initialize (modified in-place)
        gain: Multiplicative constant for standard deviation adjustment
             (default 1.0, use 5/3 for tanh, sqrt(2) for ReLU)

    Returns:
        The initialized tensor (same as input)

    Example:
        >>> weight = Tensor(np.empty((10, 20)))
        >>> xavier_normal_(weight)
        >>> # weight now contains normally distributed values

        >>> # For different activations
        >>> xavier_normal_(weight, gain=5.0/3.0)  # For tanh
    """
    fan_in, fan_out = calculate_fan_in_fan_out(tensor)
    std = gain * np.sqrt(2 / (fan_in + fan_out))
    arr = _as_np(tensor)
    _set_data(tensor, np.random.normal(0, std, arr.shape).astype(arr.dtype, copy=False))


# =============================================================================
# Kaiming/He Initialization
# =============================================================================

def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> None:
    """
    Initialize tensor with Kaiming uniform distribution (in-place).

    Initializes tensor with values drawn from U[-limit, limit] where:
    limit = gain * sqrt(3 / fan) where fan = fan_in or fan_out
    and gain = sqrt(2 / (1 + a^2))

    Args:
        tensor: Tensor to initialize (modified in-place)
        a: Negative slope for leaky ReLU (0.0 for regular ReLU)
        mode: 'fan_in' preserves magnitude in forward pass,
              'fan_out' preserves magnitude in backward pass
        nonlinearity: Name of nonlinearity ('relu', 'leaky_relu', etc.)

    Returns:
        The initialized tensor (same as input)

    Raises:
        ValueError: If mode not in ['fan_in', 'fan_out']

    Example:
        >>> weight = Tensor(np.empty((64, 32, 3, 3)))  # Conv2d weights
        >>> kaiming_uniform_(weight)
        >>> # weight now contains ReLU-appropriate values

        >>> # For leaky ReLU with slope 0.1
        >>> kaiming_uniform_(weight, a=0.1)
    """
    fan_in, fan_out = calculate_fan_in_fan_out(tensor)
    if mode == "fan_in":
        fan = fan_in
    else:
        fan = fan_out
    if nonlinearity == "relu":
        a = 0.0
    gain = np.sqrt(2 / (1 + a**2))
    limit = gain * np.sqrt(3 / fan)
    arr = _as_np(tensor)
    _set_data(tensor, np.random.uniform(-limit, limit, arr.shape).astype(arr.dtype, copy=False))


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> None:
    """
    Initialize tensor with Kaiming normal distribution (in-place).

    Initializes tensor with values drawn from N(0, std) where:
    std = gain / sqrt(fan) where fan = fan_in or fan_out
    and gain = sqrt(2 / (1 + a^2))

    Args:
        tensor: Tensor to initialize (modified in-place)
        a: Negative slope for leaky ReLU (0.0 for regular ReLU)
        mode: 'fan_in' preserves magnitude in forward pass,
              'fan_out' preserves magnitude in backward pass
        nonlinearity: Name of nonlinearity ('relu', 'leaky_relu', etc.)

    Returns:
        The initialized tensor (same as input)

    Example:
        >>> weight = Tensor(np.empty((64, 32, 3, 3)))
        >>> kaiming_normal_(weight)
        >>> # weight follows N(0, std) with Kaiming variance

        >>> # For LeakyReLU
        >>> kaiming_normal_(weight, a=0.01)
    """
    fan_in, fan_out = calculate_fan_in_fan_out(tensor)
    if mode == "fan_in":
        fan = fan_in
    else:
        fan = fan_out
    if nonlinearity == "relu":
        a = 0.0
    gain = np.sqrt(2 / (1 + a**2))
    std = gain * np.sqrt(1 / fan)
    arr = _as_np(tensor)
    _set_data(tensor, np.random.normal(0, std, arr.shape).astype(arr.dtype, copy=False))


# =============================================================================
# Normal and Uniform Initialization
# =============================================================================

def normal_(tensor: Tensor, mean: float = 0.0, std: float = 0.01) -> None:
    """
    Initialize tensor with normal (Gaussian) distribution (in-place).

    Initializes tensor with values drawn from N(mean, std^2).

    Args:
        tensor: Tensor to initialize (modified in-place)
        mean: Mean of normal distribution (default 0.0)
        std: Standard deviation (default 1.0)

    Returns:
        The initialized tensor (same as input)

    Example:
        >>> weight = Tensor(np.empty((64, 32)))
        >>> normal_(weight, mean=0.0, std=0.01)
        >>> # weight now contains values from N(0, 0.01^2)
    """
    arr = _as_np(tensor)
    _set_data(tensor, np.random.normal(mean, std, arr.shape).astype(arr.dtype, copy=False))


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> None:
    """
    Initialize tensor with uniform distribution (in-place).

    Initializes tensor with values drawn from U[a, b].

    Args:
        tensor: Tensor to initialize (modified in-place)
        a: Lower bound (default 0.0)
        b: Upper bound (default 1.0)

    Returns:
        The initialized tensor (same as input)

    Example:
        >>> weight = Tensor(np.empty((64, 32)))
        >>> uniform_(weight, a=-0.1, b=0.1)
        >>> # weight now contains values from U[-0.1, 0.1]
    """
    arr = _as_np(tensor)
    _set_data(tensor, np.random.uniform(a, b, arr.shape).astype(arr.dtype, copy=False))


def zeros_(tensor: Tensor) -> None:
    """
    Initialize tensor with zeros (in-place).

    Args:
        tensor: Tensor to initialize (modified in-place)

    Returns:
        The initialized tensor (same as input)

    Example:
        >>> bias = Tensor(np.empty((64,)))
        >>> zeros_(bias)
        >>> # bias is now all zeros
    """
    tensor.data.fill(0.0)


def ones_(tensor: Tensor) -> None:
    """
    Initialize tensor with ones (in-place).

    Args:
        tensor: Tensor to initialize (modified in-place)

    Returns:
        The initialized tensor (same as input)

    Example:
        >>> scale = Tensor(np.empty((64,)))
        >>> ones_(scale)
        >>> # scale is now all ones
    """
    tensor.data.fill(1.0)


def constant_(tensor: Tensor, value: float) -> None:
    """
    Initialize tensor with a constant value (in-place).

    Args:
        tensor: Tensor to initialize (modified in-place)
        value: Constant value to use

    Returns:
        The initialized tensor (same as input)

    Example:
        >>> bias = Tensor(np.empty((64,)))
        >>> constant_(bias, 0.1)
        >>> # bias is now all 0.1
    """
    tensor.data.fill(value)


# =============================================================================
# Orthogonal Initialization
# =============================================================================

def orthogonal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Initialize tensor as orthogonal matrix (in-place).

    Uses QR decomposition to generate orthogonal matrix:
    1. Fill tensor with random values
    2. Compute QR decomposition
    3. Q is orthogonal
    4. Scale by gain

    Args:
        tensor: Tensor to initialize (modified in-place)
                Should be at least 2D. For non-square matrices,
                orthogonality applies to the appropriate dimension.
        gain: Multiplicative scaling factor (default 1.0)

    Returns:
        The initialized tensor (same as input)

    Raises:
        ValueError: If tensor has less than 2 dimensions

    Note:
        For tensors with shape (rows, cols):
        - If rows < cols: rows are orthonormal
        - If rows >= cols: cols are orthonormal

    Example:
        >>> weight = Tensor(np.empty((512, 512)))  # RNN recurrent weight
        >>> orthogonal_(weight)
        >>> # weight is now orthogonal

        >>> # Verify orthogonality
        >>> assert np.allclose(weight.data @ weight.data.T, np.eye(512), atol=1e-6)

        >>> # For very deep networks, use gain < 1 to prevent blow-up
        >>> orthogonal_(weight, gain=0.5)
    """
    arr = _as_np(tensor)
    shape = arr.shape
    if len(shape) < 2:
        raise ValueError("Orthogonal initialization requires at least 2D tensor")

    rows = shape[0]
    cols = 1
    for s in shape[1:]:
        cols *= s

    # Generate random matrix and compute QR decomposition
    a = np.random.normal(0.0, 1.0, (rows, cols))
    if rows < cols:
        q, r = np.linalg.qr(a.T)
        q = q.T
    else:
        q, r = np.linalg.qr(a)

    # Fix signs to make decomposition unique
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph

    # Scale by gain and reshape back
    _set_data(tensor, (gain * q).reshape(shape).astype(arr.dtype, copy=False))
    return tensor
