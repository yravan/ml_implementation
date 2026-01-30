"""
Loss functions with gradients: MSE, Cross-Entropy, Binary Cross-Entropy.

Each function returns both the loss value and the gradient w.r.t. the
model's output (predictions), ready for backpropagation.

MSE:
    L = (1/N) Σ_i ||y_pred_i - y_true_i||²
    ∂L/∂y_pred = (2/N)(y_pred - y_true)

Cross-Entropy (with softmax):
    L = -(1/N) Σ_i log(softmax(logits_i)[y_true_i])
    ∂L/∂logits = (1/N)(softmax(logits) - one_hot(y_true))

Binary Cross-Entropy:
    L = -(1/N) Σ_i [y_i log(p_i) + (1-y_i) log(1-p_i)]
    ∂L/∂p = (1/N)(-y/p + (1-y)/(1-p))
"""

import numpy as np


def mse_loss(y_pred, y_true):
    """
    Mean Squared Error loss with gradient.

    Parameters:
        y_pred: np.ndarray of shape (N, D) or (N,) - Predictions.
        y_true: np.ndarray of shape (N, D) or (N,) - Targets.

    Returns:
        loss: float - Mean squared error.
        grad: np.ndarray same shape as y_pred - Gradient ∂L/∂y_pred.
    """
    loss = (y_pred - y_true) ** 2
    grad = 2 * (y_pred - y_true)
    return loss, grad


def cross_entropy_loss(logits, y_true):
    """
    Cross-entropy loss from raw logits (numerically stable softmax + NLL).

    Combines softmax + negative log likelihood:
        probs = softmax(logits)
        L = -(1/N) Σ_i log(probs[i, y_true[i]])

    The gradient has a clean form:
        ∂L/∂logits[i, j] = (1/N)(probs[i, j] - 1_{j = y_true[i]})

    Parameters:
        logits: np.ndarray of shape (N, C) - Raw scores (pre-softmax).
        y_true: np.ndarray of shape (N,) dtype int - True class indices.

    Returns:
        loss: float - Cross-entropy loss.
        grad: np.ndarray of shape (N, C) - Gradient ∂L/∂logits.
    """
    loss = -np.log(logits[y_true[:, None]]).mean()
    grad = None
    return loss, grad


def binary_cross_entropy_loss(y_pred, y_true, eps=1e-7):
    """
    Binary cross-entropy loss with gradient.

        L = -(1/N) Σ_i [y_i log(p_i) + (1 - y_i) log(1 - p_i)]

    Parameters:
        y_pred: np.ndarray of shape (N,) or (N, 1) - Predicted probabilities in (0, 1).
        y_true: np.ndarray of shape (N,) or (N, 1) - Binary labels (0 or 1).
        eps: float - Clamp value for numerical stability.

    Returns:
        loss: float - Binary cross-entropy loss.
        grad: np.ndarray same shape as y_pred - Gradient ∂L/∂y_pred.
    """
    loss = None
    grad = None
    return loss, grad
