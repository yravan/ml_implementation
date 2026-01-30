"""
Core neural network layers: Linear, ReLU, Sigmoid, Softmax, and MLP.

Each layer implements forward and backward passes for backpropagation.
The backward pass computes gradients w.r.t. both inputs and parameters.

Key notation:
    - Forward: output = f(input, params)
    - Backward: given d_output (upstream gradient), compute:
        d_input = ∂L/∂input       (to propagate further back)
        d_params = ∂L/∂params     (to update weights)

Chain rule: d_input = d_output @ ∂output/∂input
"""

import numpy as np


def linear_forward(X, W, b):
    """
    Forward pass for a fully-connected (linear) layer.

        Y = X @ W + b

    Parameters:
        X: np.ndarray of shape (batch, in_features) - Input.
        W: np.ndarray of shape (in_features, out_features) - Weights.
        b: np.ndarray of shape (out_features,) - Bias.

    Returns:
        Y: np.ndarray of shape (batch, out_features) - Output.
        cache: tuple (X, W, b) - Stored for backward pass.
    """
    Y = X @ W + b
    cache = (X, W, b)
    return Y, cache


def linear_backward(d_out, cache):
    """
    Backward pass for a fully-connected layer.

    Given d_out = ∂L/∂Y:
        ∂L/∂X = d_out @ W^T
        ∂L/∂W = X^T @ d_out
        ∂L/∂b = sum(d_out, axis=0)

    Parameters:
        d_out: np.ndarray of shape (batch, out_features) - Upstream gradient.
        cache: tuple (X, W, b) from forward pass.

    Returns:
        d_X: np.ndarray of shape (batch, in_features) - Gradient w.r.t. input.
        d_W: np.ndarray of shape (in_features, out_features) - Gradient w.r.t. weights.
        d_b: np.ndarray of shape (out_features,) - Gradient w.r.t. bias.
    """
    X, W, b = cache
    d_X = d_out @ W.T
    d_W = X.T @ d_out
    d_b = d_out.sum(axis=0)
    return d_X, d_W, d_b


def relu_forward(X):
    """
    Forward pass for ReLU activation.

        Y = max(0, X)

    Parameters:
        X: np.ndarray of any shape - Input.

    Returns:
        Y: np.ndarray same shape as X - Output.
        cache: X - Stored for backward pass.
    """
    Y = np.maximum(0, X)
    cache = X
    return Y, cache


def relu_backward(d_out, cache):
    """
    Backward pass for ReLU.

        ∂L/∂X = d_out * (X > 0)

    Parameters:
        d_out: np.ndarray - Upstream gradient.
        cache: X from forward pass.

    Returns:
        d_X: np.ndarray same shape as X - Gradient w.r.t. input.
    """
    X = cache
    d_X = np.zeros_like(X)
    d_X[X > 0] = 1
    d_X = d_X * d_out
    return d_X


def sigmoid_forward(X):
    """
    Forward pass for sigmoid activation.

        σ(x) = 1 / (1 + exp(-x))

    Parameters:
        X: np.ndarray of any shape - Input.

    Returns:
        Y: np.ndarray same shape as X - Output in (0, 1).
        cache: Y - Stored for backward (derivative uses output directly).
    """
    Y = 1 / (1 + np.exp(-X))
    cache = Y
    return Y, cache


def sigmoid_backward(d_out, cache):
    """
    Backward pass for sigmoid.

        ∂σ/∂x = σ(x)(1 - σ(x))
        ∂L/∂X = d_out * σ * (1 - σ)

    Parameters:
        d_out: np.ndarray - Upstream gradient.
        cache: Y (sigmoid output) from forward pass.

    Returns:
        d_X: np.ndarray same shape as X - Gradient w.r.t. input.
    """
    Y = cache
    d_X = Y * (1 - Y) * d_out
    return d_X


def softmax_forward(X):
    """
    Numerically stable softmax.

        softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))

    Parameters:
        X: np.ndarray of shape (batch, n_classes) - Logits.

    Returns:
        probs: np.ndarray of shape (batch, n_classes) - Probabilities (rows sum to 1).
    """
    probs = np.exp(X - np.max(X)) / np.sum(np.exp(X  - np.max(X)), axis=1, keepdims=True)
    return probs


def mlp_forward(X, params):
    """
    Forward pass through a multi-layer perceptron.

    Architecture: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear

    Parameters:
        X: np.ndarray of shape (batch, in_features) - Input.
        params: list of tuples [(W1, b1), (W2, b2), ...] - Layer parameters.
            Each (W, b) pair defines a linear layer.

    Returns:
        output: np.ndarray of shape (batch, out_features) - Final output.
        caches: list of cache tuples from each layer (for backward pass).
            Each entry is ('linear', linear_cache) or ('relu', relu_cache).
    """
    output = X
    caches = []
    for W, b in params:
        output, cache = linear_forward(output, W, b)
        caches.append(cache)
        output, cache = relu_forward(output)
        caches.append(cache)
    return output, caches


def mlp_backward(d_out, caches):
    """
    Backward pass through an MLP, computing gradients for all layers.

    Walks backward through caches, applying the chain rule at each layer.

    Parameters:
        d_out: np.ndarray of shape (batch, out_features) - Upstream gradient.
        caches: list of cache tuples from mlp_forward.

    Returns:
        d_input: np.ndarray of shape (batch, in_features) - Gradient w.r.t. input.
        grads: list of tuples [(dW1, db1), (dW2, db2), ...] - Parameter gradients.
    """
    d_input = 1
    from collections import deque
    grads = deque([])
    for c in caches[::-1]:
        if len(c) == 3:
            grad = linear_backward(d_input, c)
            d_input = grad[0]
            grads.appendleft(grad[1:])
        else:
            d_input = relu_backward(d_input, c)
    return d_input, grads
