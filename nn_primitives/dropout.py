"""
Dropout regularization.

During training, randomly sets elements to zero with probability p,
then scales surviving elements by 1/(1-p) (inverted dropout) so that
expected values are preserved without needing to scale at test time.

    mask ~ Bernoulli(1 - p)
    Y_train = X * mask / (1 - p)
    Y_test = X                     (no dropout at test time)
"""

import numpy as np


def dropout_forward(X, p=0.5, training=True, mask=None):
    """
    Forward pass for inverted dropout.

    During training:
        1. Sample binary mask: mask[i] ~ Bernoulli(1 - p)
        2. Y = X * mask / (1 - p)

    During inference:
        Y = X  (identity)

    Parameters:
        X: np.ndarray of any shape - Input.
        p: float - Dropout probability (probability of dropping a unit), in [0, 1).
        training: bool - Whether in training mode.
        mask: np.ndarray (optional) - Pre-generated mask (for reproducibility in tests).

    Returns:
        Y: np.ndarray same shape as X - Output.
        cache: tuple (mask, p) - Stored for backward.
    """
    Y = None
    cache = None
    return Y, cache


def dropout_backward(d_out, cache):
    """
    Backward pass for inverted dropout.

    Gradient only flows through positions that were not dropped:
        ∂L/∂X = d_out * mask / (1 - p)

    Parameters:
        d_out: np.ndarray - Upstream gradient.
        cache: tuple (mask, p) from forward.

    Returns:
        d_X: np.ndarray same shape as d_out - Gradient w.r.t. input.
    """
    d_X = None
    return d_X
