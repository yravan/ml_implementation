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

from python.foundations import Tensor, maximum, Function, minimum
from python.foundations.computational_graph import (
    convert_to_function,
    sigmoid,
    log,
    softmax,
)

# Import _no_grad from computational_graph for gradient tracking control
try:
    from python.foundations.computational_graph import _no_grad
except ImportError:
    _no_grad = False


# =============================================================================
# Functional Interfaces
# =============================================================================

class ReLU(Function):
    def forward(self, x:np.ndarray) -> np.ndarray:
        global _no_grad;
        if not _no_grad:
            self.mask = x <= 0
        return np.maximum(x, 0)
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        dx = grad_output.copy()
        dx[self.mask] = 0.0
        return dx,

class LeakyReLU(Function):
    def forward(self, x:np.ndarray, alpha: float = 0.01) -> np.ndarray:
        global _no_grad;
        if not _no_grad:
            self.alpha = alpha
            self.mask = x < 0
        out = x.copy()
        out[x < 0] *= alpha
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        dx = grad_output.copy()
        dx[self.mask] *= self.alpha
        return dx,

class ELU(Function):
    def forward(self, x:np.ndarray, alpha: float = 0.01) -> np.ndarray:
        global _no_grad;
        out = x.copy()
        mask = x <= 0
        out[mask] = alpha * (np.exp(x[mask]) - 1)
        if not _no_grad:
            self.alpha = alpha
            self.mask = mask
            self.out = out
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        dx = grad_output.copy()
        dx[self.mask] *= (self.out[self.mask] + self.alpha)
        return dx,


class SELU(Function):
    alpha = 1.6732632
    scale = 1.05070098
    def forward(self, x:np.ndarray) -> np.ndarray:
        global _no_grad;
        out = x.copy()
        mask = x <= 0
        out[mask] = self.alpha * (np.exp(x[mask]) - 1)
        out *= self.scale
        if not _no_grad:
            self.mask = mask
            self.out = out
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        dx = grad_output.copy()
        dx[~self.mask] *= self.scale
        dx[self.mask] *= (self.out[self.mask] + self.alpha * self.scale)
        return dx,



class PReLU(Function):
    """Parametric ReLU: max(0, x) + alpha * min(0, x)"""
    # alpha is singular or num_channels
    def forward(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        global _no_grad;
        out = x.copy()
        mask = x <= 0
        if alpha.size > 1:
            channel_dim = np.argmax(np.array(x.shape) == alpha.size)
            channel_dim = int(channel_dim)
            broadcast_alpha = alpha.reshape(((1,) * channel_dim + (-1,) + (1,) * (x.ndim - 1 - channel_dim)))
            broadcast_alpha = np.broadcast_to(broadcast_alpha, out.shape)
        else:
            broadcast_alpha = np.broadcast_to(alpha, out.shape)
        out[mask] *= broadcast_alpha[mask]
        if not _no_grad:
            self.channel_dim = channel_dim
            self.mask = mask
            self.x = x
            self.alpha = alpha
        return out


    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        dx = grad_output.copy()
        dx[self.mask] *= self.alpha
        dalpha = grad_output[self.mask] * self.x[self.mask]
        if self.alpha.size > 1:
            all_axes = tuple(range(dalpha.ndim))
            axes_to_sum = tuple(ax for ax in all_axes if ax != self.channel_dim)
            dalpha = np.sum(dalpha, axis=axes_to_sum)
        else:
            dalpha = np.sum(dalpha)
        return dx, dalpha


def relu6(x: Tensor) -> Tensor:
    return minimum(maximum(0, x), 6)

class GELU(Function):
    """
    Gaussian Error Linear Unit: x * Φ(x)

    Exact:       GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    Approximate: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    def forward(self, x: np.ndarray, approximate: bool = True) -> np.ndarray:
        global _no_grad
        self.approximate = approximate

        if approximate:
            # Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            inner = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
            self.tanh_val = np.tanh(inner)
            out = 0.5 * x * (1 + self.tanh_val)
        else:
            # Exact: x * Φ(x) where Φ is CDF of standard normal
            from scipy.special import erf
            self.cdf = 0.5 * (1 + erf(x / np.sqrt(2)))
            out = x * self.cdf

        if not _no_grad:
            self.x = x
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        x = self.x

        if self.approximate:
            # d/dx[0.5 * x * (1 + tanh(inner))] where inner = sqrt(2/π) * (x + 0.044715 * x³)
            # = 0.5 * (1 + tanh) + 0.5 * x * sech²(inner) * d(inner)/dx
            # d(inner)/dx = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
            sech2 = 1 - self.tanh_val**2
            inner_deriv = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
            dx = 0.5 * (1 + self.tanh_val) + 0.5 * x * sech2 * inner_deriv
        else:
            # d/dx[x * Φ(x)] = Φ(x) + x * φ(x)
            # where φ(x) = exp(-x²/2) / sqrt(2π) is the PDF
            pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
            dx = self.cdf + x * pdf

        return grad_output * dx,


def quickgelu(x: Tensor, scale: float = 1.702) -> Tensor:
    return x * sigmoid(x * scale)

def silu(x: Tensor) -> Tensor:
    return x * sigmoid(x)

def hard_sigmoid(x: Tensor, slope: float = 1/6, offset: float = 0.5) -> Tensor:
    return (x * slope + offset).clamp(0, 1)

def tanh(x: Tensor) -> Tensor:
    pos = x.exp()
    neg = (-x).exp()
    return (pos - neg) / (pos + neg)

def hardtanh(x: Tensor, min_val: float = -1.0, max_val: float = 1.0) -> Tensor:
    return x.clamp(min_val, max_val)

def tanhshrink(x: Tensor) -> Tensor:
    return x - tanh(x)

def softplus(x:Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    mask = (beta * x) > threshold
    out = log(1 + np.exp(beta * x)) * 1/beta
    out.set(mask, x)
    return out

def softsign(x: Tensor) -> Tensor:
    return x / (1 + x.abs())

def mish(x: Tensor) -> Tensor:
    return x * tanh(softplus(x))


# =============================================================================
# Composite / Special Activation Functions
# =============================================================================

def temperature_softmax(x: Tensor, temperature: float = 1.0,
                        axis: int = -1) -> Tensor:
    """
    Softmax with temperature scaling.

    softmax(x/T) where:
    - T < 1: sharper distribution (more confident)
    - T = 1: standard softmax
    - T > 1: softer distribution (more uniform)

    Args:
        x: Input logits
        temperature: Temperature parameter (must be > 0)
        axis: Axis for softmax

    Returns:
        Temperature-scaled probabilities
    """
    return softmax(x / temperature, axis=axis)
def gumbel_softmax(logits: np.ndarray, temperature: float = 1.0, hard: bool = False, axis: int = -1) -> np.ndarray:
    rng = np.random.default_rng()
    samples = rng.gumbel(size=logits.shape)
    out = softmax((logits + samples) / temperature, axis = axis)
    if hard:
        indices = out.argmax(axis = axis, keepdims = True)
        hard_out = out.copy()
        hard_out.fill(0.0)
        hard_out.set(indices, 1.0)
        out = hard_out - out.detach() + out
    return out


relu = convert_to_function(ReLU)
leaky_relu = convert_to_function(LeakyReLU)
elu = convert_to_function(ELU)
selu = convert_to_function(SELU)
prelu = convert_to_function(PReLU)
gelu = convert_to_function(GELU)
quickgelu = quickgelu
silu = silu
hard_sigmoid = hard_sigmoid
tanh = tanh
hardtanh = hardtanh
tanhshrink = tanhshrink
sigmoid = sigmoid
softplus = softplus
softsign = softsign
mish = mish
softmax = softmax
temperature_softmax = temperature_softmax
gumbel_softmax = gumbel_softmax
