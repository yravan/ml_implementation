"""
Computational Graph
===================

A framework for building and executing computational graphs that enable automatic differentiation.

Theory
------
Deep learning computations can be represented as Directed Acyclic Graphs (DAGs) where:
- Nodes represent operations (add, multiply, matrix multiply, activation functions)
- Edges represent data flow (tensors flowing between operations)

During the **forward pass**, we compute outputs and store intermediate values.
During the **backward pass**, we traverse the graph in reverse topological order,
applying the chain rule to compute gradients of the loss with respect to all parameters.

This is the foundation of backpropagation. PyTorch's autograd, TensorFlow's eager mode,
and JAX's autodiff all work on similar principles.

Key concepts:
1. **Tensor with gradient tracking**: Wraps NumPy arrays, tracks operations applied
2. **Operation nodes**: Functions that know how to compute forward and backward
3. **Topological sorting**: Order nodes so all dependencies come before dependents
4. **Gradient accumulation**: Sum gradients when a tensor is used multiple times

Math
----
# Chain rule for scalar functions:
# If y = f(g(x)), then dy/dx = (dy/dg) * (dg/dx)

# For vector/matrix functions (Jacobian-vector products):
# If y = f(x) where x ∈ R^n, y ∈ R^m, then:
#   Jacobian J = ∂y/∂x has shape (m, n)
#   J_ij = ∂y_i/∂x_j
#
# In backprop, we compute vector-Jacobian products (VJPs):
#   If L is a scalar loss, we have grad_y = ∂L/∂y (shape m)
#   Then grad_x = J^T @ grad_y = Σ_i (∂y_i/∂x) * (∂L/∂y_i)

# Example: y = Wx + b
#   Forward: y = Wx + b
#   Backward given ∂L/∂y:
#     ∂L/∂W = ∂L/∂y @ x^T
#     ∂L/∂x = W^T @ ∂L/∂y
#     ∂L/∂b = ∂L/∂y

Algorithm
---------
Forward Pass:
    for each operation in topological order:
        output = operation.forward(inputs)
        store output for backward pass

Backward Pass:
    grad_outputs[loss_node] = 1.0  # dL/dL = 1
    for each operation in reverse topological order:
        grad_inputs = operation.backward(grad_outputs[output])
        accumulate grad_inputs to corresponding input nodes

References
----------
- "Automatic Differentiation in Machine Learning: a Survey" (Baydin et al., 2018)
  https://arxiv.org/abs/1502.05767
- Karpathy's micrograd: Minimal autodiff implementation
  https://github.com/karpathy/micrograd
- CS231n: Backpropagation lecture
  https://cs231n.github.io/optimization-2/
- PyTorch Autograd tutorial
  https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

Implementation Notes
--------------------
- Store references to parents/children for graph traversal
- Use weak references to avoid memory leaks with circular dependencies
- Accumulate gradients (don't overwrite) when tensor used multiple times
- Clear cached values after backward to free memory
- Consider using __slots__ for memory efficiency in Tensor class
"""
from contextlib import contextmanager

# Implementation Status: NOT STARTED
# Complexity: Hard
# Prerequisites: None (foundational module)

import numpy as np
from typing import List, Optional, Tuple, Callable, Set, Union

# Import Function base class and all operations from functionals
from .functionals import (
    Function,
    # Basic arithmetic (implemented)
    Add,
    Mul,
    MatMul,
    Pow,
    Sum,
    Exp,
    Log,
    Reshape,
    Transpose,
    Max,
    # Activations (stubs)
    Softmax,
    # Shape ops (stubs)
    Concat,
    Stack,
    Split,
    Slice,
    Mean,
    Var,
    # Neural network ops (stubs)
    # Internal
    Abs,
    Clamp,
    LogSigmoid,
    Sigmoid,
    Identity,
    LogSoftmax,
    _no_grad,
    Sub,
    Neg,
    Div,
    Min,
)



def convert_to_function(cls):
    fn = cls()
    def f(*class_args, **class_kwargs):
        class_args = list(class_args)
        children = []
        for i, arg in enumerate(class_args):
            if isinstance(arg, Tensor):
                children.append(arg)
                class_args[i] = arg.data
            elif isinstance(arg, np.ndarray):
                if arg.dtype == np.float64:
                    class_args[i] = arg.astype(np.float32, copy=False)
            elif isinstance(arg, float):
                class_args[i] = np.float32(arg)
        for k, v in class_kwargs.items():
            if isinstance(v, Tensor):
                children.append(v)
                class_kwargs[k] = v.data
            elif isinstance(v, np.ndarray):
                if v.dtype == np.float64:
                    class_kwargs[k] = v.astype(np.float32, copy=False)
            elif isinstance(v, float):
                class_kwargs[k] = np.float32(v)
        requires_grad = any(arg.requires_grad for arg in children)
        out = Tensor(data=fn.forward(*class_args, **class_kwargs),
                     requires_grad=requires_grad,
                     _children=tuple(children),
                     _grad_fn=fn)
        return out
    return f


class Tensor:
    """
    A tensor that tracks its computational history for automatic differentiation.

    This is similar to PyTorch's Tensor with requires_grad=True.

    Attributes:
        data: The underlying NumPy array
        grad: Gradient of loss with respect to this tensor (computed during backward)
        requires_grad: Whether to track gradients for this tensor
        _grad_fn: The operation that created this tensor (None for leaf tensors)
        _children: Tensors that were inputs to the operation that created this
        is_leaf: True if this tensor was created by user (not by an operation)

    Example:
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        >>> y = x * 2
        >>> z = y.sum()
        >>> z.backward()
        >>> x.grad
        array([2., 2., 2.])  # dz/dx = d(sum(2*x))/dx = 2

    PyTorch Analogy:
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        y = x * 2
        z = y.sum()
        z.backward()
        x.grad  # tensor([2., 2., 2.])
    """

    def __init__(self, data: Union[np.ndarray, float, list],
                 requires_grad: bool = False,
                 _children: Tuple['Tensor', ...] = (),
                 _grad_fn: Optional['Function'] = None):
        """
        Initialize a Tensor.

        Args:
            data: The numerical data (will be converted to np.ndarray)
            requires_grad: Whether to compute gradients for this tensor
            _children: Parent tensors in the computational graph (internal use)
            _grad_fn: The function that created this tensor (internal use)
        """
        if not isinstance(data, np.ndarray):
            self.data = np.asarray(data, dtype=np.float32)
        else:
            self.data = data if data.dtype == np.float32 else data.astype(np.float32)
        global _no_grad
        self.requires_grad = not _no_grad and requires_grad
        self._children = _children
        self._grad_fn = _grad_fn
        self.is_leaf = _grad_fn is None
        self.grad: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return shape of underlying data."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return self.data.ndim

    @property
    def dtype(self):
        """Data type."""
        return self.data.dtype

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.data.size

    def numpy(self) -> np.ndarray:
        """Return the underlying numpy array."""
        return self.data

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """
        Compute gradients via backpropagation.

        This traverses the computational graph in reverse topological order,
        computing gradients using the chain rule.

        Args:
            grad: Gradient of loss w.r.t. this tensor. If None and tensor is scalar,
                  defaults to 1.0 (for loss.backward()).

        Raises:
            RuntimeError: If called on non-scalar tensor without grad argument

        Example:
            >>> x = Tensor([1.0, 2.0], requires_grad=True)
            >>> y = (x ** 2).sum()  # scalar
            >>> y.backward()  # No grad needed for scalar
            >>> x.grad
            array([2., 4.])
        """
        if self.grad is None:
            if self.ndim == 0:
                self.grad = np.array([1.0], dtype=self.dtype)
            else:
                self.grad = np.ones_like(self.data, dtype=self.data.dtype)
        else:
            if grad.ndim != self.ndim:
                raise RuntimeError("Expected grad with the same dim")
            if grad.shape != self.shape:
                raise RuntimeError("Expected grad with the same shape")

        # ── Iterative topological sort ──
        topo = []
        visited = set()
        stack = [(self, False)]

        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))  # push "emit me after my children"
            for child in node._children:
                if child not in visited:
                    stack.append((child, False))

        # ── Backward pass ──
        for node in reversed(topo):
            if node.requires_grad and node._grad_fn is not None:
                child_grads = node._grad_fn.backward(node.grad)
                for child, child_grad in zip(node._children, child_grads):
                    if child.requires_grad:
                        if child.grad is None:
                            child.grad = child_grad
                        else:
                            child.grad += child_grad

    def zero_grad(self) -> None:
        """Zero out gradients. Call before each backward pass in training."""
        self.grad = None

    # ==================== Arithmetic Operations ====================
    # Each operation creates a new Tensor with appropriate _grad_fn

    def __add__(self, other: Union['Tensor', float, np.ndarray]) -> 'Tensor':
        """Element-wise addition: self + other."""
        add = convert_to_function(Add)
        return add(self, other)

    def __radd__(self, other: Union[float, np.ndarray]) -> 'Tensor':
        """Reverse addition for scalar + Tensor."""
        add = convert_to_function(Add)
        return add(other, self)

    def __mul__(self, other: Union['Tensor', float, np.ndarray]) -> 'Tensor':
        """Element-wise multiplication: self * other."""
        mul = convert_to_function(Mul)
        return mul(self, other)

    def __rmul__(self, other: Union[float, np.ndarray]) -> 'Tensor':
        """Reverse multiplication."""
        mul = convert_to_function(Mul)
        return mul(other, self)

    def __neg__(self) -> 'Tensor':
        """Negation: -self."""
        neg = convert_to_function(Neg)
        return neg(self)

    def __sub__(self, other: Union['Tensor', float, np.ndarray]) -> 'Tensor':
        """Subtraction: self - other."""
        sub = convert_to_function(Sub)
        return sub(self, other)

    def __rsub__(self, other: Union[float, np.ndarray]) -> 'Tensor':
        """Reverse subtraction."""
        sub = convert_to_function(Sub)
        return sub(Tensor(other), self)

    def __truediv__(self, other: Union['Tensor', float, np.ndarray]) -> 'Tensor':
        """Division: self / other."""
        div = convert_to_function(Div)
        if isinstance(other, (int, np.integer)):
            other = float(other)
        return div(self, other)

    def __rtruediv__(self, other: Union[float, np.ndarray]) -> 'Tensor':
        """Reverse division."""
        div = convert_to_function(Div)
        return div(Tensor(other), self)

    def __pow__(self, power: float) -> 'Tensor':
        """Power: self ** power."""
        pow = convert_to_function(Pow)
        return pow(self, power)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication: self @ other."""
        matmul = convert_to_function(MatMul)
        return matmul(self, other)

    def abs(self) -> 'Tensor':
        """Absolute sum: self + other."""
        abs = convert_to_function(Abs)
        return abs(self)

    def clamp(self, min_val: Optional[Union[float, np.ndarray, int, 'Tensor']] = 0, max_val: Optional[Union[float, np.ndarray, int, 'Tensor']] = 1) -> 'Tensor':
        clamp = convert_to_function(Clamp)
        return clamp(self, min_val, max_val)

    def __ge__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(np.greater_equal(self.data, other.data))
        else:
            return Tensor(np.greater_equal(self.data, other))

    def __gt__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(np.greater(self.data, other.data))
        else:
            return Tensor(np.greater(self.data, other))

    def __le__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(np.less_equal(self.data, other.data))
        else:
            return Tensor(np.less_equal(self.data, other))

    def __lt__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(np.less(self.data, other.data))
        else:
            return Tensor(np.less(self.data, other))

    def __invert__(self) -> 'Tensor':
        # if not self.dtype == np.bool_:
        #     raise RuntimeError('Cannot invert non bool tensor')
        return Tensor(~(self.data).astype(np.bool))

    def copy(self) -> 'Tensor':
        identity = convert_to_function(Identity)
        return identity(self)

    def detach(self) -> 'Tensor':
        out = self.copy()
        out.requires_grad = False
        out.grad = None
        return out

    def argmax(self, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> 'Tensor':
        arg = np.argmax(self.data, axis=axis, keepdims=keepdims)
        return Tensor(arg)

    # ==================== Reduction Operations ====================

    def sum(self, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> 'Tensor':
        """Sum elements, optionally along an axis."""
        sum = convert_to_function(Sum)
        return sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> 'Tensor':
        """Mean of elements."""
        mean = convert_to_function(Mean)
        return mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> 'Tensor':
        """Max of elements or element-wise max with another tensor."""
        max = convert_to_function(Max)
        return max(self, axis=axis, keepdims=keepdims)

    def min(self, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> 'Tensor':
        """Min of elements or element-wise max with another tensor."""
        min = convert_to_function(Min)
        return min(self, axis=axis, keepdims=keepdims)
    def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """Variance of elements."""
        var = convert_to_function(Var)
        return var(self, axis=axis, keepdims=keepdims)

    # ==================== Shape Operations ====================

    def reshape(self, *shape: int) -> 'Tensor':
        """Reshape tensor."""
        reshape = convert_to_function(Reshape)
        return reshape(self, shape)

    def transpose(self, *axes: int) -> 'Tensor':
        """Transpose tensor."""
        transpose = convert_to_function(Transpose)
        return transpose(self, *axes)

    def split(self, indices_or_sections, axis:int = 0) -> 'Tensor':
        split = convert_to_function(Split)
        return split(self, indices_or_sections, axis=axis)

    def fill(self, value: float) -> 'Tensor':
        self.data[:] = value
        self.requires_grad = False
        self._children = None
        return self

    def set_in_place(self, indices: Union[int, Tuple[int], List[int], np.ndarray, 'Tensor'], value: Union[float, np.ndarray, 'Tensor']) -> 'Tensor':
        if isinstance(indices, Tensor):
            indices = indices.data
        if isinstance(value, Tensor):
            self.requires_grad = self.requires_grad or value.requires_grad
            value = value.data
            self._children = self._children + (value,)
        else:
            self.requires_grad = False
            self._children = None
        self.data[indices] = value
        return self

    def set(self, indices: Union[int, Tuple[int], List[int], np.ndarray, 'Tensor'], values: Union[float, np.ndarray, 'Tensor']) -> 'Tensor':
        set_fn = convert_to_function(Set)
        return set_fn(self, indices, values)

    @property
    def T(self) -> 'Tensor':
        """Matrix transpose (last two dimensions)."""
        return self.transpose()

    def __getitem__(self, slices) -> 'Tensor':
        """Indexing/slicing."""
        slice = convert_to_function(Slice)
        if isinstance(slices, Tensor):
            slices = tuple(slices.data)
        return slice(self, slices)

    def sigmoid(self) -> 'Tensor':
        """Sigmoid activation."""
        sigmoid = convert_to_function(Sigmoid)
        return sigmoid(self)

    def log_sigmoid(self) -> 'Tensor':
        """Logarithm sigmoid activation."""
        logsigmoid = convert_to_function(LogSigmoid)
        return logsigmoid(self)

    def exp(self) -> 'Tensor':
        """Exponential."""
        exp = convert_to_function(Exp)
        return exp(self)

    def log(self) -> 'Tensor':
        """Natural logarithm."""
        log = convert_to_function(Log)
        return log(self)

    def softmax(self, axis: int = -1) -> 'Tensor':
        """Softmax activation."""
        softmax = convert_to_function(Softmax)
        return softmax(self, axis=axis)

    def log_softmax(self, axis: int = -1) -> 'Tensor':
        """Logarithm softmax activation."""
        logsoftmax = convert_to_function(LogSoftmax)
        return logsoftmax(self, axis=axis)


    def __repr__(self) -> str:
        """String representation."""
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"


def maximum(*inputs: Union[Tensor, np.ndarray, float, int]) -> Tensor:
    inputs = list(inputs)
    for i, input in enumerate(inputs):
        if not isinstance(input, Tensor):
            inputs[i] = Tensor(input)
    middle = stack(*inputs)
    return middle.max(axis=0)

def minimum(*inputs: Union[Tensor, np.ndarray, float, int]) -> Tensor:
    inputs = list(inputs)
    for i, input in enumerate(inputs):
        if not isinstance(input, Tensor):
            inputs[i] = Tensor(input)
    middle = stack(*inputs)
    return -(-middle).max(axis=0)

def stack(*inputs: Union[Tensor, np.ndarray, float, int], axis: int = 0) -> Tensor:
    inputs = list(inputs)
    for i, input in enumerate(inputs):
        if not isinstance(input, Tensor):
            inputs[i] = Tensor(input)
    fn = Stack()
    data_inputs = [i.data for i in inputs]
    broadcasted = np.broadcast_arrays(*data_inputs)
    return Tensor(fn.forward(*broadcasted, axis=axis),
                  requires_grad=any([i.requires_grad for i in inputs]),
                  _children=tuple(inputs),
                  _grad_fn=fn,
    )
def concat(*inputs: Union[Tensor, np.ndarray, float, int], axis: int = 0) -> Tensor:
    concat_fn = convert_to_function(Concat)
    return concat_fn(*inputs, axis=axis)


def print_graph(tensor, indent=0, visited=None):
    """Print the computation graph for a tensor."""
    if visited is None:
        visited = {}

    prefix = "  " * indent
    grad_fn_name = type(tensor._grad_fn).__name__ if tensor._grad_fn else "Leaf"
    grad_str = "✓" if tensor.requires_grad else "✗"

    node_id = id(tensor)
    seen = node_id in visited
    visited[node_id] = True

    print(
        f"{prefix}[{grad_str}] {grad_fn_name} → shape={tensor.shape} dtype={tensor.dtype}"
        + (" (seen)" if seen else "")
    )

    if seen or not tensor._children:
        return

    for child in tensor._children:
        print_graph(child, indent + 1, visited)

# ==================== Context Manager ====================

@contextmanager
def no_grad():
    """
    Context manager to disable gradient tracking.

    Useful for evaluation/inference when you don't need gradients.

    Example:
        >>> with no_grad():
        ...     y = model(x)  # No gradient tracking
    """
    global _no_grad
    old_value = _no_grad
    try:
        _no_grad = True
        yield
    finally:
        _no_grad = old_value

