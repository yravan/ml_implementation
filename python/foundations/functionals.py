"""
Functional Operations (Stateful Functions for Autograd)
=======================================================

This module contains all the stateful Function classes that implement
differentiable operations for the computational graph.

Each Function class:
- Inherits from the abstract Function base class
- Implements forward() which computes the operation and stores state
- Implements backward() which computes gradients using stored state

The Tensor class uses these Functions to build the computational graph.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

from python.utils.math_utils import sigmoid, softmax, log_sigmoid, softplus, log_softmax


class Function(ABC):
    """
    Base class for differentiable functions (operations).

    Each Function subclass implements:
    - forward: Compute output from inputs
    - backward: Compute gradients given upstream gradient

    The Function stores any values needed for backward pass (e.g., inputs).

    Example subclass:
        class ReLU(Function):
            def forward(self, x):
                self.mask = x > 0
                return np.maximum(x, 0)

            def backward(self, grad_output):
                return (grad_output * self.mask,)
    """

    @abstractmethod
    def forward(self, *inputs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute forward pass.

        Store any values needed for backward.

        Args:
            *inputs: Input arrays

        Returns:
            Output array
        """
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Compute backward pass.

        Args:
            grad_output: Gradient of loss w.r.t. output of forward()

        Returns:
            Tuple of gradients w.r.t. each input to forward()
        """
        pass

    def __call__(self, *inputs: np.ndarray) -> np.ndarray:
        return self.forward(*inputs)



# ==================== Helper Functions ====================

def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...], func: str = "sum") -> np.ndarray:
    """
    Sum gradient over dimensions that were broadcast.

    When a tensor with shape (3,) is added to one with shape (2, 3),
    the result has shape (2, 3). In backward, we need to sum over
    the first axis to get gradient for the (3,) tensor.
    """
    grad_shape = np.array(grad.shape)
    target_shape_pad = (1,) * (len(grad_shape) - len(target_shape)) + target_shape
    target_shape_pad = np.array(target_shape_pad)
    broadcast_dims = np.where((target_shape_pad != grad_shape) & (grad_shape > 1))[0]
    broadcast_dims = tuple(broadcast_dims)
    if broadcast_dims:
        if func == "sum":
            grad = grad.sum(axis=broadcast_dims, keepdims=True)
        elif func == "mean":
            grad = grad.mean(axis=broadcast_dims, keepdims=True)
    return grad.reshape(target_shape).copy()


# Global flag for disabling gradient computation
_no_grad = False


# ==================== Basic Arithmetic Operations (IMPLEMENTED) ====================

class Identity(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.copy()
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        return grad_output.copy(),


class Set(Function):
    def forward(self, x: np.ndarray, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
        x[indices] = values
        global _no_grad
        if not _no_grad:
            self.indices = indices
        return x.copy()

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        grad_x = grad_output.copy()
        grad_x[self.indices] = 0
        return grad_x,

class Add(Function):
    """
    Element-wise addition: z = x + y.

    Forward: z = x + y
    Backward: ∂L/∂x = ∂L/∂z, ∂L/∂y = ∂L/∂z

    Broadcasting is handled by summing over broadcasted dimensions.
    """

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute x + y."""
        global _no_grad
        if not _no_grad:
            self.x = x
            self.y = y
        return x + y

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients for addition."""
        dx = grad_output
        dy = grad_output
        return _unbroadcast(dx, self.x.shape), _unbroadcast(dy, self.y.shape)


class Mul(Function):
    """
    Element-wise multiplication: z = x * y.

    Forward: z = x * y
    Backward: ∂L/∂x = ∂L/∂z * y, ∂L/∂y = ∂L/∂z * x
    """

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute x * y."""
        global _no_grad
        if not _no_grad:
            self.x = x
            self.y = y
        return x * y

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients: ∂L/∂x = grad * y, ∂L/∂y = grad * x."""
        dx = grad_output * self.y
        dy = grad_output * self.x
        return _unbroadcast(dx, self.x.shape), _unbroadcast(dy, self.y.shape)


class MatMul(Function):
    """
    Matrix multiplication: Z = X @ Y.

    For X: (m, n) and Y: (n, p), output Z: (m, p)

    Backward:
    ∂L/∂X = ∂L/∂Z @ Y^T  (shape: (m, p) @ (p, n) = (m, n))
    ∂L/∂Y = X^T @ ∂L/∂Z  (shape: (n, m) @ (m, p) = (n, p))
    """

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute X @ Y."""
        global _no_grad
        if not _no_grad:
            self.x = x
            self.y = y
        return x @ y

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients for matrix multiplication."""
        dx = grad_output @ self.y.T
        dy = self.x.T @ grad_output
        return dx, dy


class Pow(Function):
    """
    Power operation: z = x ** p (p is a constant).

    Forward: z = x^p
    Backward: ∂L/∂x = ∂L/∂z * p * x^(p-1)
    """
    def forward(self, x: np.ndarray, power: float) -> np.ndarray:
        """Compute x ** power."""
        global _no_grad
        if not _no_grad:
            self.power = power
            self.x = x
        return x ** power

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Compute gradient: p * x^(p-1) * grad."""
        dx = self.power * (self.x ** (self.power - 1)) * grad_output
        return (dx,)


class Sum(Function):
    """
    Sum reduction: y = sum(x, axis).

    Backward: gradient is broadcast back to original shape.
    """

    def forward(self, x: np.ndarray, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> np.ndarray:
        """Compute sum along axis."""
        global _no_grad
        if not _no_grad:
            self.axis = axis
            self.keepdims = keepdims
            self.x = x
            self.shape = x.shape
        return np.sum(x, axis=axis, keepdims=keepdims)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Broadcast gradient back to input shape."""
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)
        return (np.broadcast_to(grad_output, self.shape),)

class Clamp(Function):
    def forward(self, x: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
        """Compute exp(x)."""
        global _no_grad
        if not _no_grad:
            self.mask = (x >= min_val) & (x <= max_val)
        return np.clip(x, min_val, max_val)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient of exp is exp."""
        dx = np.zeros_like(grad_output)
        dx[self.mask] = grad_output[self.mask]
        return dx,


class Exp(Function):
    """
    Exponential: z = exp(x).

    Forward: z = e^x
    Backward: ∂L/∂x = ∂L/∂z * e^x = ∂L/∂z * z
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute exp(x)."""
        self.output = np.exp(x)
        return self.output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient of exp is exp."""
        return (grad_output * self.output,)


class Log(Function):
    """
    Natural logarithm: z = log(x).

    Forward: z = ln(x)
    Backward: ∂L/∂x = ∂L/∂z * (1/x)
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute log(x)."""
        global _no_grad
        if not _no_grad:
            self.x = x
        return np.log(x)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient of log is 1/x."""
        return (grad_output / self.x,)


class Reshape(Function):
    """Reshape operation that preserves gradient flow."""

    def forward(self, x: np.ndarray, new_shape: Tuple[int, ...]) -> np.ndarray:
        global _no_grad
        if not _no_grad:
            self.old_shape = x.shape
            self.new_shape = new_shape
        return x.reshape(new_shape)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        return (grad_output.reshape(self.old_shape),)


class Transpose(Function):
    """Transpose operation."""

    def forward(self, x: np.ndarray, axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        global _no_grad
        if not _no_grad:
            self.axes = axes
            self.x = x
        return x.transpose(axes)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        if self.axes is None:
            return (grad_output.T,)
        inverse_axes = np.argsort(self.axes)
        return (grad_output.transpose(inverse_axes),)


class Max(Function):
    """
    Max reduction: y = max(x, axis).

    Backward: gradient flows only to max elements.
    """

    def forward(self, x: np.ndarray, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> np.ndarray:
        """Compute max along axis."""
        global _no_grad
        if not _no_grad:
            self.x = x
            self.shape = x.shape
            self.axis = axis
            self.keepdims = keepdims

        return np.max(x, axis=axis, keepdims=keepdims)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient flows to max elements."""
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)

        max_vals = np.max(self.x, axis=self.axis, keepdims=True)
        mask = (self.x == max_vals).astype(np.uint8)
        mask = mask / mask.sum(axis=self.axis, keepdims=True)

        return (mask * grad_output,)


class Abs(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        global _no_grad
        if not _no_grad:
            self.mask = x < 0
        return np.abs(x)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        dx = grad_output.copy()
        dx[self.mask] *= -1
        return (dx,)


class Sigmoid(Function):
    """
    Sigmoid activation: y = 1 / (1 + exp(-x)).

    Forward: y = σ(x) = 1 / (1 + e^(-x))
    Backward: ∂L/∂x = ∂L/∂y * y * (1 - y)
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = sigmoid(x)
        global _no_grad
        if not _no_grad:
            self.out = out
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        dx = self.out * (1 - self.out) * grad_output
        return dx,

class LogSigmoid(Function):
    """
    Sigmoid activation: y = 1 / (1 + exp(-x)).
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = log_sigmoid(x)
        global _no_grad
        if not _no_grad:
            self.out = out
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        dx = (1 - np.exp(self.out)) * grad_output
        return dx,


class Softmax(Function):
    """
    Softmax activation: y_i = exp(x_i) / sum(exp(x_j)).

    Forward: y = softmax(x) along axis
    Backward: Jacobian is y_i * (δ_ij - y_j), applied as VJP
    """

    def forward(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        global _no_grad
        out = softmax(x, axis)
        if not _no_grad:
            self.out = out
            self.axis = axis
        return self.out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        sum_term = np.sum(grad_output * self.out, axis=self.axis, keepdims=True)
        return (self.out * (grad_output - sum_term),)

class LogSoftmax(Function):
    def forward(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        global _no_grad
        out = log_softmax(x, axis)
        if not _no_grad:
            self.out = out
            self.axis = axis
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        # d(log_softmax)/dx = I - softmax
        # VJP: dx = grad - softmax * sum(grad)
        softmax = np.exp(self.out)
        sum_term = np.sum(grad_output, axis=self.axis, keepdims=True)
        return (grad_output - softmax * sum_term,)


# ==================== Shape/Data Operations (STUBS) ====================

class Concat(Function):
    """
    Concatenate tensors along an axis.

    Forward: y = concat([x1, x2, ...], axis)
    Backward: Split gradient back to original shapes
    """
    def forward(self, *inputs: np.ndarray, axis: int = 0) -> np.ndarray:
        out = np.concatenate(inputs, axis=axis)
        global _no_grad
        if not _no_grad:
            self.axis = axis
            self.split_indices = np.cumsum([x.shape[self.axis] for x in inputs[:-1]])
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        grads = np.split(grad_output, self.split_indices, axis=self.axis)
        return tuple(grads)

class Stack(Function):
    """
    Stack tensors along a new axis.

    Forward: y = stack([x1, x2, ...], axis)
    Backward: Unstack gradient
    """
    def forward(self, *inputs: np.ndarray, axis: int = 0) -> np.ndarray:
        global _no_grad
        if not _no_grad:
            self.axis = axis
        return np.stack(inputs, axis=axis)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        num_inputs = grad_output.shape[self.axis]
        grads = np.split(grad_output, num_inputs, axis=self.axis)
        return tuple(g.squeeze(axis=self.axis) for g in grads)


class Split(Function):
    """
    Split tensor along an axis.

    Forward: [y1, y2, ...] = split(x, indices_or_sections, axis)
    Backward: Concatenate gradients
    """
    def forward(self, x: np.ndarray, indices_or_sections, axis: int = 0) -> List[np.ndarray]:
        global _no_grad;
        if not _no_grad:
            self.indices_or_sections = indices_or_sections
            self.axis = axis
        return np.split(x, indices_or_sections, axis=axis)

    def backward(self, *grad_outputs: np.ndarray) -> Tuple[np.ndarray]:
        return np.concatenate(grad_outputs, axis=self.axis),



class Slice(Function):
    """
    Slice/index operation for tensors.

    Forward: y = x[slices]
    Backward: Scatter gradient back to original positions
    """
    def forward(self, x: np.ndarray, slices) -> np.ndarray:
        global _no_grad
        if not _no_grad:
            self.slices = slices
            self.shape = x.shape
        return x[slices]

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        grad = np.zeros(self.shape)
        grad[self.slices] = grad_output
        return grad,

class Mean(Function):
    def forward(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> np.ndarray:
        global _no_grad
        mean = np.mean(x, axis=axis, keepdims=keepdims)
        if not _no_grad:
            self.axis = axis
            self.keepdims = keepdims
            self.shape = x.shape
            self.count = x.size // mean.size
        return mean

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)
        grad_output = np.broadcast_to(grad_output, self.shape)
        return grad_output / self.count,

class Var(Function):
    """
    Variance reduction: y = var(x, axis).

    Math: var(x) = mean((x - mean(x))^2)
    Backward: ∂L/∂x = ∂L/∂y * 2 * (x - mean(x)) / n
    """

    def forward(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> np.ndarray:
        global _no_grad
        var = np.var(x, axis=axis, keepdims=keepdims)
        if not _no_grad:
            self.axis = axis
            self.keepdims = keepdims
            self.mean = np.mean(x, axis=self.axis, keepdims=True)
            self.count = x.size // var.size
            self.x = x
        return var

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)
        return grad_output * 2 * (self.x - self.mean) / self.count,

