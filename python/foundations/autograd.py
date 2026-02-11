"""
Automatic Differentiation (Reverse Mode)
========================================

High-level automatic differentiation engine building on the computational graph.

Theory
------
Automatic differentiation (autodiff) computes exact gradients (not numerical approximations)
by systematically applying the chain rule. There are two modes:

1. **Forward mode**: Compute derivatives alongside values. Good when #outputs >> #inputs.
   - Computes Jacobian-vector products (JVPs): J @ v
   - One forward pass per input variable

2. **Reverse mode**: Compute derivatives backward from output. Good when #inputs >> #outputs.
   - Computes vector-Jacobian products (VJPs): v^T @ J
   - One backward pass per output variable
   - This is what deep learning uses (one loss, many parameters)

Deep learning uses reverse mode because:
- Loss is scalar (1 output)
- Parameters can be millions (many inputs)
- One backward pass gives all gradients

The backward pass is also called "backpropagation" in neural networks.

Math
----
# Given a computation: L = f(g(h(x)))
#
# Forward: compute values
#   a = h(x)
#   b = g(a)
#   L = f(b)
#
# Backward: apply chain rule in reverse
#   ∂L/∂b = f'(b)           # Local gradient * upstream gradient (1)
#   ∂L/∂a = g'(a) * ∂L/∂b   # Local gradient * upstream gradient
#   ∂L/∂x = h'(x) * ∂L/∂a   # Local gradient * upstream gradient
#
# Key insight: each node only needs:
#   1. Its inputs (stored during forward)
#   2. The upstream gradient (∂L/∂output)
# To compute: downstream gradients (∂L/∂inputs)

# For a general DAG (not just chain):
# If y = f(x1, x2, ..., xn), and L = g(y, ...):
#   ∂L/∂xi = Σ_j (∂L/∂yj * ∂yj/∂xi)
# The sum handles the case where xi affects multiple yj

Algorithm
---------
1. Forward pass:
   - Evaluate operations in topological order
   - Store intermediate values for backward

2. Build backward graph:
   - Find all nodes between leaves and loss
   - Topologically sort them

3. Backward pass:
   - Initialize ∂L/∂L = 1
   - For each node in reverse topological order:
     - Get upstream gradient
     - Compute local gradients using stored values
     - Pass gradients to children
     - Accumulate if a tensor is used multiple times

References
----------
- "Automatic Differentiation in Machine Learning: a Survey"
  https://arxiv.org/abs/1502.05767
- "Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation"
  (Griewank & Walther) - The textbook on autodiff
- JAX documentation on autodiff
  https://jax.readthedocs.io/en/latest/autodidax.html
- "Calculus on Computational Graphs: Backpropagation" - Chris Olah
  https://colah.github.io/posts/2015-08-Backprop/

Implementation Notes
--------------------
- Use a tape/trace to record operations (dynamic computation graphs)
- Handle in-place operations carefully (they can break gradient computation)
- Gradient checkpointing trades memory for compute (recompute during backward)
- Detach tensors from graph when you don't need gradients
"""

# Implementation Status: NOT STARTED
# Complexity: Hard
# Prerequisites: computational_graph

import numpy as np
from typing import List, Callable, Dict, Tuple, Optional, Set, Any
from .computational_graph import Tensor, Function

# _gradient_tape: bool = False
# class GradientTape:
#     """
#     Records operations for automatic differentiation.
#
#     This is similar to TensorFlow's GradientTape. Operations are recorded
#     when the tape is active, then gradients are computed with tape.gradient().
#
#     Example:
#         >>> x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
#         >>> with GradientTape() as tape:
#         ...     y = x ** 2
#         ...     loss = y.sum()
#         >>> grads = tape.gradient(loss, [x])
#         >>> grads[0]  # ∂loss/∂x = 2*x
#         array([2., 4., 6.])
#
#     Advantages:
#         - Explicit control over what's recorded
#         - Can compute gradients multiple times (if persistent=True)
#         - Clear memory management
#
#     PyTorch Note:
#         PyTorch uses implicit tape (records automatically when requires_grad=True).
#         TensorFlow and JAX use explicit tapes like this.
#     """
#
#     def __init__(self, persistent: bool = False):
#         """
#         Initialize GradientTape.
#
#         Args:
#             persistent: If True, tape can be used multiple times.
#                        If False (default), tape is cleared after gradient().
#         """
#         self.persistent = persistent
#         self.operations: List[Tuple[Function, Tuple[Tensor,...], Tensor]] = []
#         self.watching = set()
#         self.gradient_table: Optional[Dict[Tensor, np.ndarray]] = None
#
#     def __enter__(self) -> 'GradientTape':
#         """Start recording."""
#         global _gradient_tape
#         _gradient_tape = self
#         return self  # Must return self for "with GradientTape() as tape:" syntax
#
#     def __exit__(self, *args: Any) -> None:
#         """Stop recording."""
#         global _gradient_tape
#         _gradient_tape = None
#
#     def watch(self, tensor: Tensor) -> None:
#         """
#         Explicitly watch a tensor.
#
#         Tensors with requires_grad=True are automatically watched.
#         Use this for tensors that don't require grad but you want derivatives for.
#
#         Args:
#             tensor: Tensor to track
#         """
#         self.watching.add(tensor)
#
#     def record(self, func: Function, inputs: Tuple[Tensor, ...],
#                output: Tensor) -> None:
#         """
#         Record an operation (called internally by Tensor operations).
#
#         Args:
#             func: The function/operation
#             inputs: Input tensors
#             output: Output tensor
#         """
#         self.operations.append((func, inputs, output))
#
#     def gradient(self, target: Tensor, sources: List[Tensor],
#                  output_gradients: Optional[np.ndarray] = None
#                  ) -> List[Optional[np.ndarray]]:
#         """
#         Compute gradients of target with respect to sources.
#
#         Args:
#             target: The tensor to differentiate (usually loss)
#             sources: Tensors to compute gradients for (usually parameters)
#             output_gradients: Gradient of some upstream loss w.r.t. target.
#                              Defaults to 1.0 (for scalar target).
#
#         Returns:
#             List of gradients, one per source. None if no path from source to target.
#
#         Example:
#             >>> with GradientTape() as tape:
#             ...     y = model(x)
#             ...     loss = loss_fn(y, labels)
#             >>> grads = tape.gradient(loss, model.parameters())
#             >>> for param, grad in zip(model.parameters(), grads):
#             ...     param.data -= learning_rate * grad
#         """
#         if self.gradient_table is None:
#             self.gradient_table = {}
#         if output_gradients is None:
#             if target.ndim == 0:
#                 output_gradients = 1.0
#             else:
#                 raise RuntimeError("gradient computation only supported for scalar target")
#         else:
#             if output_gradients.ndim != target.ndim:
#                 raise RuntimeError("gradient computation only supported for scalar target")
#             if output_gradients.shape != target.shape:
#                 raise RuntimeError("gradient computation only supported for scalar target")
#         self.gradient_table[target] = output_gradients
#         for fn, inputs, output in self.operations[::-1]:
#             input_grads = fn.backward(inputs, self.gradient_table[output])
#             del(self.gradient_table[output])
#             for param, grad in zip(inputs, input_grads):
#                 if param not in self.gradient_table:
#                     self.gradient_table[param] = grad
#                 else:
#                     self.gradient_table[param] += grad
#         gradients = [self.gradient_table[s] for s in sources]
#         if not self.persistent:
#             self.gradient_table = None
#         return gradients


class Variable(Tensor):
    """
    A learnable parameter with automatic gradient accumulation.

    Variables are the weights and biases of neural networks.
    They track their gradients across multiple backward passes
    until explicitly zeroed.

    Attributes:
        data: The parameter values (NumPy array)
        grad: Accumulated gradients (None until first backward)
        requires_grad: Always True for Variables

    Example:
        >>> W = Variable(np.random.randn(784, 256) * 0.01)
        >>> b = Variable(np.zeros(256))
        >>> # In training loop:
        >>> W.zero_grad()
        >>> b.zero_grad()
        >>> # ... forward and backward ...
        >>> W.data -= lr * W.grad
        >>> b.data -= lr * b.grad
    """

    def __init__(self, data: np.ndarray, name: str = ""):
        """
        Initialize a Variable.

        Args:
            data: Initial parameter values
            name: Optional name for debugging
        """
        super().__init__(data, requires_grad=True)
        self.name = name

    def __repr__(self) -> str:
        return f"Variable({self.shape}, name={self.name})"


def grad(func: Callable, argnums: int = 0) -> Callable:
    """
    Create a function that computes gradients.

    This is a functional API for autodiff, similar to JAX's grad.

    Args:
        func: Function to differentiate. Must return a scalar.
        argnums: Which argument to differentiate with respect to.

    Returns:
        Function that computes gradient of func w.r.t. specified argument.

    Example:
        >>> def f(x):
        ...     return (x ** 2).sum()
        >>> grad_f = grad(f)
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> grad_f(x)
        array([2., 4., 6.])

    JAX-style:
        >>> from jax import grad
        >>> grad(lambda x: x**2)(3.0)
        6.0
    """
    def grad_f(*args):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, Tensor):
                arg.requires_grad = True
        out = func(*args)
        out.backward()
        if isinstance(argnums, int):
            gradients = args[argnums].grad
        else:
            gradients = tuple(args[i].grad for i in argnums)
        return gradients
    return grad_f

def value_and_grad(func: Callable, argnums: int = 0
                   ) -> Callable[..., Tuple[Any, np.ndarray]]:
    """
    Create function that returns both value and gradient.

    More efficient than computing them separately.

    Args:
        func: Function to differentiate
        argnums: Which argument to differentiate

    Returns:
        Function returning (value, gradient)

    Example:
        >>> def loss(x, y):
        ...     return ((x - y) ** 2).sum()
        >>> val_and_grad = value_and_grad(loss)
        >>> x = np.array([1.0, 2.0])
        >>> y = np.array([0.0, 0.0])
        >>> loss_val, grad_x = val_and_grad(x, y)
        >>> loss_val
        5.0
        >>> grad_x
        array([2., 4.])
    """
    def grad_f(*args):
        args = list(args)
        for i, arg in enumerate(args):
            if not isinstance(arg, Tensor):
                args[i] = Tensor(arg, requires_grad=True)
            else:
                arg.requires_grad = True
        out = func(*args)
        out.backward()
        if isinstance(argnums, int):
            gradients = (args[argnums].grad,)
        else:
            gradients = tuple(args[i].grad for i in argnums)
        return (out,) + gradients
    return grad_f

def jacobian(func: Callable, x: np.ndarray) -> np.ndarray:
    """
    Compute full Jacobian matrix.

    For f: R^n -> R^m, the Jacobian J is an (m, n) matrix where
    J[i, j] = ∂f_i/∂x_j.

    Note: This is expensive for large outputs. Each row requires
    a backward pass (reverse mode) or each column requires a forward pass.

    Args:
        func: Function R^n -> R^m
        x: Input point, shape (n,)

    Returns:
        Jacobian matrix, shape (m, n)

    Example:
        >>> def f(x):
        ...     return np.array([x[0]**2, x[0]*x[1], x[1]**2])
        >>> jacobian(f, np.array([1.0, 2.0]))
        array([[2., 0.],
               [2., 1.],
               [0., 4.]])
    """
    y = func(x)
    J = np.zeros((y.shape[0], x.shape[0]))
    one_hot = np.zeros_like(y)
    for i in range(y.shape[0]):
        one_hot.fill(0); one_hot[i] = 1
        x.zero_grad()
        (y * one_hot).backward()
        J[i, :] = x.grad()
    return J


def hessian(func: Callable, x: np.ndarray) -> np.ndarray:
    """
    Compute Hessian matrix (matrix of second derivatives).

    For f: R^n -> R, the Hessian H is an (n, n) matrix where
    H[i, j] = ∂²f/∂x_i∂x_j.

    Computed by taking Jacobian of the gradient.

    Args:
        func: Scalar function R^n -> R
        x: Input point

    Returns:
        Hessian matrix, shape (n, n)

    Example:
        >>> def f(x):
        ...     return x[0]**2 + x[0]*x[1] + x[1]**2
        >>> hessian(f, np.array([1.0, 1.0]))
        array([[2., 1.],
               [1., 2.]])
    """
    grad_fn = grad(func)
    hessian = jacobian(grad_fn, x)
    return hessian

def jvp(func: Callable, primals: Tuple[np.ndarray, ...],
        tangents: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jacobian-vector product (forward-mode autodiff).

    Computes (f(x), J @ v) where J is the Jacobian of f at x.

    This is forward-mode autodiff: compute derivative alongside value.
    Efficient when output dim >> input dim.

    Args:
        func: Function to differentiate
        primals: Input values (x)
        tangents: Tangent vectors (v)

    Returns:
        Tuple of (f(x), J @ v)

    Example:
        >>> def f(x):
        ...     return x ** 2
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> v = np.array([1.0, 0.0, 0.0])  # Direction
        >>> y, jvp_val = jvp(f, (x,), (v,))
        >>> jvp_val  # Directional derivative in direction v
        array([2., 0., 0.])  # = 2*x * v
    """
    J = jacobian(func, primals[0])
    return func(primals), J @ tangents


def vjp(func: Callable, x: np.ndarray) -> Tuple[np.ndarray, Callable]:
    """
    Vector-Jacobian product setup (reverse-mode autodiff).

    Returns the function value and a function to compute VJPs.

    Args:
        func: Function to differentiate
        x: Input value

    Returns:
        Tuple of (f(x), vjp_func) where vjp_func(v) computes v^T @ J

    Example:
        >>> def f(x):
        ...     return x ** 2
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> y, vjp_fn = vjp(f, x)
        >>> vjp_fn(np.ones(3))  # Gradient when upstream grad is all 1s
        array([2., 4., 6.])
    """
    J = jacobian(func, x)
    def vjp_func(v):
        return J @ v
    return func(x), vjp_func


# ==================== Gradient Transformations ====================

def stop_gradient(x: Tensor) -> Tensor:
    """
    Stop gradient propagation through this tensor.

    The value passes through unchanged, but gradients are blocked.
    Useful for:
    - Detaching targets in self-supervised learning
    - Implementing straight-through estimators
    - Preventing gradient flow in certain paths

    Args:
        x: Input tensor

    Returns:
        Tensor with same value but no gradient tracking

    Example:
        >>> x = Tensor([1.0, 2.0], requires_grad=True)
        >>> y = x * 2
        >>> z = stop_gradient(y) + x  # Gradient only through second x
        >>> z.backward(np.ones(2))
        >>> x.grad
        array([1., 1.])  # Not [3., 3.] because y path is blocked
    """
    return Tensor(x.data, requires_grad=False)


def checkpoint(func: Callable, *args: Tensor) -> Tensor:
    """
    Gradient checkpointing for memory efficiency.

    During forward: compute and discard intermediate activations.
    During backward: recompute intermediates when needed.

    Trades compute for memory. Useful for very deep networks
    or large batch sizes.

    Args:
        func: Function to checkpoint
        *args: Input tensors

    Returns:
        Output tensor with checkpointed backward

    Example:
        >>> # Without checkpointing: stores all intermediate activations
        >>> y = layer1(x)
        >>> y = layer2(y)
        >>> y = layer3(y)
        >>>
        >>> # With checkpointing: recomputes during backward
        >>> y = checkpoint(layer1, x)
        >>> y = checkpoint(layer2, y)
        >>> y = checkpoint(layer3, y)
    """
    raise NotImplementedError(
        "TODO: Implement gradient checkpointing\n"
        "Hint:\n"
        "  - Forward: run func, don't store intermediates\n"
        "  - Backward: re-run func to get intermediates, then backprop"
    )
