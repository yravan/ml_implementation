"""
Foundations Module
==================

Core infrastructure for automatic differentiation and gradient computation.

This module provides:
- Computational graph for tracking operations
- Reverse-mode automatic differentiation (backpropagation)
- Gradient checking utilities for verification

Start here! Everything in deep learning depends on these foundations.
"""

# Core Tensor class
from .computational_graph import Tensor, no_grad, stack, concat, maximum, minimum, convert_to_function

# All Function classes from functionals
from .functionals import (
    # Base class
    Function,
    # Basic arithmetic (implemented)
    Add, Mul, MatMul, Pow, Sum, Exp, Log, Reshape, Transpose, Max,
    # Activations (stubs)
    Sigmoid, Softmax,
    # Shape ops (stubs)
    Concat, Stack, Split, Slice, Mean, Var,

    _no_grad,
)

# Autograd utilities
from .autograd import Variable, grad, value_and_grad

# Gradient checking
from .gradient_check import numerical_gradient, gradient_check, gradcheck, GradientChecker, GradCheckReport

__all__ = [
    # Core
    'Tensor', 'Function', 'maximum', 'minimum', 'stack', 'concat',
    # Basic ops
    'Add', 'Mul', 'MatMul', 'Pow', 'Sum', 'Exp', 'Log', 'Reshape', 'Transpose', 'Max',
    # Activations
    'Sigmoid', 'Softmax',
    # Shape ops
    'Concat', 'Stack', 'Split', 'Slice', 'Mean', 'Var',
    # Context managers
    'no_grad',
    # Autograd
    'Variable', 'grad', 'value_and_grad',
    # Gradient Checking
    'numerical_gradient', 'gradient_check', 'gradcheck',
]
