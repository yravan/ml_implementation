"""
Linear (Fully Connected) Layer
==============================

The fundamental building block of neural networks: y = Wx + b.

Theory
------
A linear layer (also called dense or fully connected) performs an affine transformation:
    y = Wx + b

where:
- x ∈ R^{batch × in_features}: input Tensor
- W ∈ R^{out_features × in_features}: weight Parameter
- b ∈ R^{out_features}: bias Parameter
- y ∈ R^{batch × out_features}: output Tensor

Linear layers:
1. Transform input dimensions (e.g., 784 -> 256)
2. Learn linear combinations of features
3. Are universal approximators when stacked with nonlinearities

Without nonlinear activations between them, stacking linear layers is equivalent
to a single linear layer (composition of linear functions is linear).

Math
----
# Forward pass:
# y = x @ W.T + b
# or equivalently: y_i = Σ_j W_ij * x_j + b_i

# Backward pass is handled automatically by the computational graph!
# When using Tensors with requires_grad=True:
# ∂L/∂x = ∂L/∂y @ W         # Computed by matmul backward
# ∂L/∂W = (∂L/∂y).T @ x     # Computed by matmul backward
# ∂L/∂b = sum(∂L/∂y, axis=0) # Computed by add backward

Note: Unlike the old np.ndarray approach, modules NO LONGER need backward()
methods. The Tensor class and Function classes handle gradient computation
automatically through the computational graph.

References
----------
- CS231n: Neural Networks Part 1
  https://cs231n.github.io/neural-networks-1/
- "Understanding Deep Learning" Ch. 4: Deep Networks
  https://udlbook.github.io/udlbook/
- PyTorch Linear documentation
  https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
"""

# Implementation Status: STUB
# Complexity: Easy
# Prerequisites: foundations/computational_graph, foundations/functionals

import numpy as np
from typing import Optional, Tuple, Union

from .init import normal_
from .module import Module, Parameter
from python.foundations import Tensor

class Linear(Module):
    """
    Linear transformation: y = xW^T + b

    A fully connected layer that applies a linear transformation to input data.
    Takes Tensor inputs and returns Tensor outputs with automatic gradient
    computation via the computational graph.

    Attributes:
        weight: Weight Parameter of shape (out_features, in_features)
        bias: Bias Parameter of shape (out_features,) or None

    Example:
        >>> linear = Linear(784, 256)  # MNIST -> hidden
        >>> x = Tensor(np.random.randn(32, 784), requires_grad=True)  # batch of 32
        >>> y = linear(x)
        >>> y.data.shape
        (32, 256)
        >>> # Backward is automatic when calling y.backward()!
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = 'normal'
    ):
        """
        Initialize Linear layer.

        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If True, add a learnable bias
            init: Initialization method ('xavier', 'kaiming', 'normal', 'zeros')

        Initialization:
            - Xavier: Good for tanh/sigmoid activations
              W ~ Uniform(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
            - Kaiming: Good for ReLU activations
              W ~ Normal(0, √(2/fan_in))
        """
        super().__init__()
        self.weight = Parameter(np.zeros((out_features, in_features)))
        if bias:
            self.bias = Parameter(np.zeros((out_features,)))
        else:
            self.bias = None
        if init == 'xavier':
            raise NotImplementedError
        elif init == 'kaiming':
            raise NotImplementedError
        elif init == 'normal':
            normal_(self.weight, std=0.01)
            normal_(self.bias, std=0.01)
        elif init == 'zeros':
            pass
        else:
            raise ValueError(f"Unknown init method: {init}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = xW^T + b

        Args:
            x: Input Tensor of shape (batch_size, in_features)
               Can also be (..., in_features) for arbitrary batch dims

        Returns:
            Output Tensor of shape (batch_size, out_features)

        Note:
            Since we use Tensor operations (matmul, add), the backward
            pass is handled automatically by the computational graph.
            No need for a backward() method!
        """
        out = x @ self.weight.T
        if self.bias is not None: out += self.bias
        return out

    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Bilinear(Module):
    """
    Bilinear transformation: y = x1^T W x2 + b

    Applies a bilinear transformation to two inputs.
    Output[k] = x1^T @ W[k] @ x2 + b[k]

    Used in:
    - Attention mechanisms
    - Relation modeling
    - Low-rank approximations

    Attributes:
        weight: Parameter tensor of shape (out_features, in1_features, in2_features)
        bias: Parameter vector of shape (out_features,)

    Example:
        >>> bilinear = Bilinear(64, 32, 10)
        >>> x1 = Tensor(np.random.randn(8, 64))  # (batch, in1)
        >>> x2 = Tensor(np.random.randn(8, 32))  # (batch, in2)
        >>> y = bilinear(x1, x2)  # (batch, out)
    """

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        init: str = 'xavier'
    ):
        """
        Initialize Bilinear layer.

        Args:
            in1_features: Size of first input
            in2_features: Size of second input
            out_features: Size of output
            bias: Whether to include bias
        """
        super().__init__()
        self.weight = Parameter(np.zeros((out_features, in1_features, in2_features)))
        if bias:
            self.bias = Parameter(np.zeros((out_features,)))
        else:
            self.bias = None
        if init == 'xavier':
            raise NotImplementedError
        elif init == 'kaiming':
            raise NotImplementedError
        elif init == 'normal':
            raise NotImplementedError
        elif init == 'zeros':
            pass
        else:
            raise ValueError(f"Unknown init method: {init}")

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x1: First input Tensor, shape (batch, in1_features)
            x2: Second input Tensor, shape (batch, in2_features)

        Returns:
            Output Tensor, shape (batch, out_features)

        Note:
            Backward is automatic via the computational graph.
        """
        x1 = x1.reshape(*x1.shape[:-1], 1, -1)
        x2 = x2.reshape(*x1.shape[:-1], 1, -1)
        out = (((x1 @ self.weight) * x2).sum(axis=-1))
        if self.bias is not None: out += self.bias
        return out

    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return (
            f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )


class LazyLinear(Module):
    """
    Linear layer with lazy initialization.

    The in_features is inferred from the first input, making it easier
    to build networks without manually calculating intermediate sizes.

    Example:
        >>> lazy = LazyLinear(256)  # out_features only
        >>> x = Tensor(np.random.randn(32, 784))  # first call infers in_features=784
        >>> y = lazy(x)  # Creates weight (256, 784) on first call
    """

    def __init__(self, out_features: int, bias: bool = True, init: str = 'xavier'):
        """
        Initialize LazyLinear layer.

        Args:
            out_features: Size of each output sample
            bias: If True, add a learnable bias
            init: Initialization method
        """
        super().__init__()
        self.out_features = out_features
        if bias:
            self.bias = Parameter(np.zeros((out_features,)))
        else:
            self.bias = None
        self.init = init
        self.in_features = None
        self.weight = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with lazy initialization."""
        self.in_features = x.shape[-1]
        if self.weight is None:
            self._initialize_parameters()
        out = x @ self.weight.T
        if self.bias is not None: out += self.bias
        return out

    def _initialize_parameters(self):
        """Initialize weight and bias after in_features is known."""
        self.weight = Parameter(np.zeros((self.out_features, self.in_features)))
        if self.init == 'xavier':
            raise NotImplementedError
        elif self.init == 'kaiming':
            raise NotImplementedError
        elif self.init == 'normal':
            raise NotImplementedError
        elif self.init == 'zeros':
            pass
        else:
            raise ValueError(f"Unknown init method: {self.init}")

    def extra_repr(self) -> str:
        in_str = self.in_features if self.in_features else "?"
        return f"in_features={in_str}, out_features={self.out_features}, bias={self.use_bias}"

