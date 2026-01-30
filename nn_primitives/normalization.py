"""
Normalization layers: BatchNorm and LayerNorm.

BatchNorm normalizes over the batch dimension (per channel):
    μ = (1/N) Σ_n x_n            (mean over batch)
    σ² = (1/N) Σ_n (x_n - μ)²   (variance over batch)
    x̂ = (x - μ) / √(σ² + ε)     (normalize)
    y = γ x̂ + β                  (scale and shift)

LayerNorm normalizes over the feature dimension (per sample):
    μ = (1/D) Σ_d x_d
    σ² = (1/D) Σ_d (x_d - μ)²
    x̂ = (x - μ) / √(σ² + ε)
    y = γ x̂ + β

BatchNorm uses running statistics at test time; LayerNorm doesn't.
"""

import numpy as np


def batch_norm_forward(X, gamma, beta, running_mean, running_var,
                       training=True, momentum=0.1, eps=1e-5):
    """
    Forward pass for batch normalization (for fully-connected layers).

    During training:
        1. Compute batch mean and variance.
        2. Normalize: x̂ = (x - μ_batch) / √(σ²_batch + ε)
        3. Scale and shift: y = γ x̂ + β
        4. Update running stats: running_mean = (1-m)*running_mean + m*μ_batch

    During inference:
        Use running_mean and running_var instead of batch statistics.

    Parameters:
        X: np.ndarray of shape (N, D) - Input (batch of N samples, D features).
        gamma: np.ndarray of shape (D,) - Learned scale parameter.
        beta: np.ndarray of shape (D,) - Learned shift parameter.
        running_mean: np.ndarray of shape (D,) - Running mean (updated in-place).
        running_var: np.ndarray of shape (D,) - Running variance (updated in-place).
        training: bool - Whether in training mode.
        momentum: float - Momentum for running stats update.
        eps: float - Small constant for numerical stability.

    Returns:
        Y: np.ndarray of shape (N, D) - Normalized output.
        cache: tuple - Stored for backward (only needed during training).
    """
    batch_mean, batch_var = X.mean(axis=0), X.var(axis=0)
    if training:
        running_mean[:] = running_mean * (momentum) + (1 - momentum) * batch_mean
        running_var[:] = running_var * (momentum) + (1 - momentum) * batch_var
    else:
        batch_mean = running_mean
        batch_var = running_var
    Y = gamma[None,:] * (X - batch_mean[None,:]) / (np.sqrt(batch_var[None,:]) + eps) + beta[None,:]
    cache = (X, running_mean, running_var, gamma, beta)
    return Y, cache


def batch_norm_backward(d_out, cache):
    """
    Backward pass for batch normalization.

    This is one of the trickiest backward passes. The key insight is that
    the mean and variance depend on ALL samples in the batch, so the
    gradient flows through them.

    Given d_out = ∂L/∂Y:
        ∂L/∂γ = Σ_n d_out_n * x̂_n
        ∂L/∂β = Σ_n d_out_n
        ∂L/∂x̂ = d_out * γ
        ∂L/∂σ² = Σ_n ∂L/∂x̂_n * (x_n - μ) * (-1/2)(σ² + ε)^{-3/2}
        ∂L/∂μ = Σ_n ∂L/∂x̂_n * (-1/√(σ²+ε)) + ∂L/∂σ² * (-2/N) Σ_n (x_n - μ)
        ∂L/∂x = ∂L/∂x̂ / √(σ²+ε) + ∂L/∂σ² * 2(x-μ)/N + ∂L/∂μ / N

    Parameters:
        d_out: np.ndarray of shape (N, D) - Upstream gradient.
        cache: tuple from batch_norm_forward.

    Returns:
        d_X: np.ndarray of shape (N, D) - Gradient w.r.t. input.
        d_gamma: np.ndarray of shape (D,) - Gradient w.r.t. scale.
        d_beta: np.ndarray of shape (D,) - Gradient w.r.t. shift.
    """
    X, running_mean, running_var, gamma, beta = cache
    d_gamma = ((X - running_mean) / running_var * d_out).sum(axis=0)
    d_beta = (d_out).sum(axis=0)
    batch_mean, batch_var = X.mean(axis=0), X.var(axis=0)

    return d_X, d_gamma, d_beta


def layer_norm_forward(X, gamma, beta, eps=1e-5):
    """
    Forward pass for layer normalization.

    Normalizes over the feature dimension (axis=-1) independently per sample.

        μ_n = (1/D) Σ_d x_{n,d}
        σ²_n = (1/D) Σ_d (x_{n,d} - μ_n)²
        x̂ = (x - μ) / √(σ² + ε)
        y = γ x̂ + β

    Parameters:
        X: np.ndarray of shape (N, D) - Input.
        gamma: np.ndarray of shape (D,) - Learned scale.
        beta: np.ndarray of shape (D,) - Learned shift.
        eps: float - Numerical stability constant.

    Returns:
        Y: np.ndarray of shape (N, D) - Normalized output.
        cache: tuple - Stored for backward.
    """
    batch_mean, batch_var = X.mean(axis=1), X.var(axis=1)
    Y = (X - batch_mean[:, None]) /( np.sqrt(batch_var[:, None])  + eps)* gamma[None, :] + beta[None,:]
    cache = None
    return Y, cache


def layer_norm_backward(d_out, cache):
    """
    Backward pass for layer normalization.

    Similar structure to batch norm backward, but normalization is over
    the feature dimension instead of the batch dimension.

    Parameters:
        d_out: np.ndarray of shape (N, D) - Upstream gradient.
        cache: tuple from layer_norm_forward.

    Returns:
        d_X: np.ndarray of shape (N, D) - Gradient w.r.t. input.
        d_gamma: np.ndarray of shape (D,) - Gradient w.r.t. scale.
        d_beta: np.ndarray of shape (D,) - Gradient w.r.t. shift.
    """
    d_X = None
    d_gamma = None
    d_beta = None
    return d_X, d_gamma, d_beta
