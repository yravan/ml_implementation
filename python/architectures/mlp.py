"""
Multi-Layer Perceptron (MLP)
============================

The simplest deep neural network: stack of linear layers with activations.

Theory
------
An MLP (also called feedforward network or fully-connected network) consists of:
1. Input layer (just passes data through)
2. Hidden layers (linear transform + nonlinear activation)
3. Output layer (linear transform, possibly with final activation)

Each hidden layer: h = activation(Wx + b)

MLPs are universal approximators: with enough hidden units and one hidden layer,
an MLP can approximate any continuous function to arbitrary precision (given
infinite data). In practice, deeper networks often work better than wide shallow ones.

Architecture choices:
- **Width**: Number of units per layer (e.g., 256, 512, 1024)
- **Depth**: Number of hidden layers (e.g., 2-5 for simple tasks)
- **Activation**: ReLU is standard, GELU for transformers
- **Normalization**: BatchNorm or LayerNorm between layers
- **Dropout**: Regularization, typically 0.1-0.5

Math
----
# Forward pass for L-layer MLP:
# h_0 = x                                    # Input
# h_l = activation(W_l @ h_{l-1} + b_l)     # Hidden layers (l = 1, ..., L-1)
# y = W_L @ h_{L-1} + b_L                    # Output (often no activation)

# Backward pass (backpropagation):
# For each layer l = L, L-1, ..., 1:
#   δ_l = activation'(z_l) * (W_{l+1}^T @ δ_{l+1})  # Error signal
#   ∂L/∂W_l = δ_l @ h_{l-1}^T                        # Weight gradient
#   ∂L/∂b_l = δ_l                                    # Bias gradient
# where z_l = W_l @ h_{l-1} + b_l (pre-activation)

# Number of parameters:
# Sum over layers: (input_dim + 1) * output_dim
# Example: MLP(784, [256, 128], 10)
#   Layer 1: (784 + 1) * 256 = 200,960
#   Layer 2: (256 + 1) * 128 = 32,896
#   Layer 3: (128 + 1) * 10 = 1,290
#   Total: 235,146 parameters

References
----------
- "Understanding Deep Learning" Ch. 4: Deep Networks
  https://udlbook.github.io/udlbook/
- "Deep Learning" Ch. 6: Deep Feedforward Networks
  https://www.deeplearningbook.org/contents/mlp.html
- Karpathy's intro to neural networks
  https://karpathy.github.io/2019/04/25/recipe/
- Universal approximation theorem
  Cybenko (1989), Hornik (1991)

Implementation Notes
--------------------
- Store all layer objects for parameter access
- Forward pass: cache activations for backward
- Backward pass: work backwards through layers
- Initialize weights properly (Xavier for tanh, Kaiming for ReLU)
- Consider adding hooks for intermediate activations
"""

# Implementation Status: NOT STARTED
# Complexity: Easy
# Prerequisites: nn_core/layers/linear, nn_core/activations/*

import numpy as np
from typing import List, Union, Optional, Callable, Tuple

from python.nn_core import Module, Parameter, Sequential, ModuleList
from python.nn_core.layers.linear import Linear
from python.nn_core.activations.relu import ReLU, LeakyReLU
from python.nn_core.activations.tanh import Tanh
from python.nn_core.activations.sigmoid import Sigmoid
from python.nn_core.activations.gelu import GELU


class MLP(Module):
    """
    Multi-Layer Perceptron for classification or regression.

    A stack of linear layers with activation functions.

    Example (MNIST classifier):
        >>> mlp = MLP(
        ...     input_dim=784,
        ...     hidden_dims=[256, 128],
        ...     output_dim=10,
        ...     activation='relu',
        ...     dropout=0.2
        ... )
        >>> x = np.random.randn(32, 784)  # Batch of 32
        >>> logits = mlp.forward(x)
        >>> logits.shape
        (32, 10)

    Training loop:
        >>> optimizer = Adam(mlp.parameters(), lr=0.001)
        >>> for epoch in range(num_epochs):
        ...     for X_batch, y_batch in dataloader:
        ...         # Forward
        ...         logits = mlp.forward(X_batch)
        ...         loss = cross_entropy(logits, y_batch)
        ...
        ...         # Backward
        ...         mlp.zero_grad()
        ...         grad = cross_entropy_backward(logits, y_batch)
        ...         mlp.backward(grad)
        ...
        ...         # Update
        ...         optimizer.step(mlp.gradients())
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 activation: str = 'relu',
                 output_activation: Optional[str] = None,
                 dropout: float = 0.0,
                 use_batchnorm: bool = False,
                 init: str = 'kaiming'):
        """
        Initialize MLP.

        Args:
            input_dim: Size of input features
            hidden_dims: List of hidden layer sizes, e.g., [256, 128]
            output_dim: Size of output (num_classes for classification)
            activation: Activation function ('relu', 'tanh', 'gelu', 'sigmoid')
            output_activation: Activation for output layer (None for logits)
            dropout: Dropout probability (0 = no dropout)
            use_batchnorm: Add BatchNorm after each hidden layer
            init: Weight initialization ('kaiming', 'xavier', 'normal')
        """
        super().__init__()
        raise NotImplementedError(
            "TODO: Initialize MLP\n"
            "Hint:\n"
            "  self.layers = ModuleList([])  # List of Linear layers\n"
            "  self.activations = []  # List of activation functions\n"
            "  self.dropouts = []  # List of Dropout layers\n"
            "  self.batchnorms = ModuleList([])  # List of BatchNorm layers (if used)\n"
            "  \n"
            "  # Build layers\n"
            "  dims = [input_dim] + hidden_dims + [output_dim]\n"
            "  \n"
            "  for i in range(len(dims) - 1):\n"
            "      # Linear layer\n"
            "      layer = Linear(dims[i], dims[i+1], init=init)\n"
            "      self.layers.append(layer)\n"
            "      \n"
            "      # Activation (except for output layer unless specified)\n"
            "      if i < len(dims) - 2:  # Hidden layers\n"
            "          self.activations.append(get_activation(activation))\n"
            "          if use_batchnorm:\n"
            "              self.batchnorms.append(BatchNorm1d(dims[i+1]))\n"
            "          if dropout > 0:\n"
            "              self.dropouts.append(Dropout(dropout))\n"
            "      else:  # Output layer\n"
            "          if output_activation:\n"
            "              self.activations.append(get_activation(output_activation))\n"
            "          else:\n"
            "              self.activations.append(None)\n"
            "  \n"
            "  self.training = True  # For dropout/batchnorm mode"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor, shape (batch_size, input_dim)

        Returns:
            Output tensor, shape (batch_size, output_dim)
        """
        raise NotImplementedError(
            "TODO: Implement MLP forward pass\n"
            "Hint:\n"
            "  self._cache = []  # Store for backward\n"
            "  h = x\n"
            "  \n"
            "  for i, layer in enumerate(self.layers):\n"
            "      # Linear\n"
            "      h = layer.forward(h)\n"
            "      \n"
            "      # Activation (if exists)\n"
            "      if self.activations[i] is not None:\n"
            "          h = self.activations[i].forward(h)\n"
            "      \n"
            "      # BatchNorm (if exists and not output layer)\n"
            "      if self.batchnorms and i < len(self.batchnorms):\n"
            "          h = self.batchnorms[i].forward(h, training=self.training)\n"
            "      \n"
            "      # Dropout (if exists and not output layer)\n"
            "      if self.dropouts and i < len(self.dropouts):\n"
            "          h = self.dropouts[i].forward(h, training=self.training)\n"
            "  \n"
            "  return h"
        )

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through all layers.

        Computes gradients for all parameters.

        Args:
            grad_output: Gradient of loss w.r.t. output, shape (batch, output_dim)

        Returns:
            Gradient of loss w.r.t. input (optional, for stacking)
        """
        raise NotImplementedError(
            "TODO: Implement MLP backward pass\n"
            "Hint:\n"
            "  grad = grad_output\n"
            "  \n"
            "  # Backward through layers in reverse order\n"
            "  for i in range(len(self.layers) - 1, -1, -1):\n"
            "      # Dropout backward (if exists)\n"
            "      if self.dropouts and i < len(self.dropouts):\n"
            "          grad = self.dropouts[i].backward(grad)\n"
            "      \n"
            "      # BatchNorm backward (if exists)\n"
            "      if self.batchnorms and i < len(self.batchnorms):\n"
            "          grad = self.batchnorms[i].backward(grad)\n"
            "      \n"
            "      # Activation backward (if exists)\n"
            "      if self.activations[i] is not None:\n"
            "          grad = self.activations[i].backward(grad)\n"
            "      \n"
            "      # Linear backward\n"
            "      grad = self.layers[i].backward(grad)\n"
            "  \n"
            "  return grad"
        )

    def parameters(self) -> List[np.ndarray]:
        """
        Get all learnable parameters.

        Returns:
            List of parameter arrays (weights and biases)
        """
        raise NotImplementedError(
            "TODO: Collect all parameters\n"
            "Hint:\n"
            "  params = []\n"
            "  for layer in self.layers:\n"
            "      params.extend(layer.parameters())\n"
            "  for bn in self.batchnorms:\n"
            "      params.extend(bn.parameters())\n"
            "  return params"
        )

    def gradients(self) -> List[np.ndarray]:
        """
        Get gradients for all parameters (after backward).

        Returns:
            List of gradient arrays, same order as parameters()
        """
        raise NotImplementedError(
            "TODO: Collect all gradients\n"
            "Hint: Same structure as parameters() but return gradients"
        )

    def zero_grad(self) -> None:
        """Zero out all gradients."""
        raise NotImplementedError(
            "TODO: Zero gradients for all layers\n"
            "Hint:\n"
            "  for layer in self.layers:\n"
            "      layer.zero_grad()"
        )

    def train(self) -> None:
        """Set model to training mode (enables dropout)."""
        self.training = True

    def eval(self) -> None:
        """Set model to evaluation mode (disables dropout)."""
        self.training = False

    def num_parameters(self) -> int:
        """Count total number of learnable parameters."""
        raise NotImplementedError(
            "TODO: return sum(p.size for p in self.parameters())"
        )

    def __repr__(self) -> str:
        """String representation of the model."""
        raise NotImplementedError(
            "TODO: Return informative string\n"
            "Example: 'MLP(784 -> 256 -> 128 -> 10, ReLU, dropout=0.2)'"
        )


class ResidualMLP(Module):
    """
    MLP with residual (skip) connections.

    Each block: h = h + MLP_block(h)

    Residual connections help with gradient flow in deeper networks.
    Requires hidden dimensions to be the same (or use projection).
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_blocks: int = 2,
                 activation: str = 'relu',
                 dropout: float = 0.0):
        """
        Initialize ResidualMLP.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension (same for all blocks for residual)
            output_dim: Output dimension
            num_blocks: Number of residual blocks
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        raise NotImplementedError(
            "TODO: Initialize ResidualMLP\n"
            "Hint:\n"
            "  # Input projection\n"
            "  self.input_proj = Linear(input_dim, hidden_dim)\n"
            "  \n"
            "  # Residual blocks\n"
            "  self.blocks = ModuleList([])\n"
            "  for _ in range(num_blocks):\n"
            "      block = Sequential(\n"
            "          Linear(hidden_dim, hidden_dim),\n"
            "          get_activation(activation),\n"
            "          Linear(hidden_dim, hidden_dim),\n"
            "      )\n"
            "      self.blocks.append(block)\n"
            "  \n"
            "  # Output projection\n"
            "  self.output_proj = Linear(hidden_dim, output_dim)"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward with residual connections."""
        raise NotImplementedError(
            "TODO: Implement ResidualMLP forward\n"
            "Hint:\n"
            "  h = self.input_proj.forward(x)\n"
            "  for block in self.blocks:\n"
            "      residual = h\n"
            "      for layer in block:\n"
            "          h = layer.forward(h)\n"
            "      h = h + residual  # Skip connection\n"
            "  return self.output_proj.forward(h)"
        )

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward through residual blocks."""
        raise NotImplementedError("TODO: Implement backward with skip connections")


def get_activation(name: str) -> object:
    """
    Get activation function by name.

    Args:
        name: One of 'relu', 'tanh', 'sigmoid', 'gelu', 'leaky_relu'

    Returns:
        Activation class instance
    """
    raise NotImplementedError(
        "TODO: Return appropriate activation\n"
        "Hint:\n"
        "  activations = {\n"
        "      'relu': ReLU,\n"
        "      'tanh': Tanh,\n"
        "      'sigmoid': Sigmoid,\n"
        "      'gelu': GELU,\n"
        "      'leaky_relu': LeakyReLU,\n"
        "  }\n"
        "  return activations[name.lower()]()"
    )


class MLPBlock(Module):
    """
    Single MLP block with optional dropout and activation.

    Components: Linear -> Activation -> Dropout (optional)
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: str = "relu", dropout: float = 0.0):
        """
        Initialize MLP block.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            activation: Activation function name
            dropout: Dropout probability
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.dropout = dropout
        self.linear = Linear(in_features, out_features)
        self.activation_fn = get_activation(activation) if activation else None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        raise NotImplementedError(
            "TODO: Implement forward pass\n"
            "Hint:\n"
            "  h = self.linear.forward(x)\n"
            "  if self.activation_fn:\n"
            "      h = self.activation_fn.forward(h)\n"
            "  return h"
        )

