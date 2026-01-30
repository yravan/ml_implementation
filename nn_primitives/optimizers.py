"""
Optimization algorithms: SGD, SGD with momentum, and Adam.

All optimizers update parameters in-place given gradients.

SGD:
    θ = θ - lr * g

SGD + Momentum:
    v = μ * v - lr * g
    θ = θ + v

Adam:
    m = β₁ m + (1 - β₁) g          (first moment)
    v = β₂ v + (1 - β₂) g²         (second moment)
    m̂ = m / (1 - β₁^t)              (bias correction)
    v̂ = v / (1 - β₂^t)
    θ = θ - lr * m̂ / (√v̂ + ε)

Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015)
"""

import numpy as np


def sgd_step(params, grads, lr):
    """
    Vanilla SGD parameter update.

        θ_new = θ - lr * ∂L/∂θ

    Parameters:
        params: list of np.ndarray - Model parameters [W1, b1, W2, b2, ...].
        grads: list of np.ndarray - Gradients, same structure as params.
        lr: float - Learning rate.

    Returns:
        params: list of np.ndarray - Updated parameters.
    """
    params = None
    return params


def sgd_momentum_step(params, grads, velocities, lr, momentum=0.9):
    """
    SGD with momentum update.

        v_new = μ * v - lr * g
        θ_new = θ + v_new

    Parameters:
        params: list of np.ndarray - Model parameters.
        grads: list of np.ndarray - Gradients.
        velocities: list of np.ndarray - Velocity terms (same structure as params).
            Initialize to zeros on first call.
        lr: float - Learning rate.
        momentum: float - Momentum coefficient μ (typically 0.9).

    Returns:
        params: list of np.ndarray - Updated parameters.
        velocities: list of np.ndarray - Updated velocities.
    """
    params = None
    velocities = None
    return params, velocities


def adam_step(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam optimizer update.

    For each parameter θ_i with gradient g_i:
        m_i = β₁ m_i + (1 - β₁) g_i
        v_i = β₂ v_i + (1 - β₂) g_i²
        m̂_i = m_i / (1 - β₁^t)
        v̂_i = v_i / (1 - β₂^t)
        θ_i = θ_i - lr * m̂_i / (√v̂_i + ε)

    Parameters:
        params: list of np.ndarray - Model parameters.
        grads: list of np.ndarray - Gradients.
        m: list of np.ndarray - First moment estimates (initialize to zeros).
        v: list of np.ndarray - Second moment estimates (initialize to zeros).
        t: int - Current timestep (starting from 1).
        lr: float - Learning rate.
        beta1: float - Exponential decay rate for first moment.
        beta2: float - Exponential decay rate for second moment.
        eps: float - Numerical stability constant.

    Returns:
        params: list of np.ndarray - Updated parameters.
        m: list of np.ndarray - Updated first moments.
        v: list of np.ndarray - Updated second moments.
    """
    params = None
    m = None
    v = None
    return params, m, v
