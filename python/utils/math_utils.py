"""
Mathematical Utilities
======================

Core mathematical functions with numerical stability guarantees.

Theory
------
Numerical stability is critical in deep learning. Operations like exp() and log()
can easily overflow or underflow with float32/float64 precision. This module provides
numerically stable implementations of common operations used throughout ML:

1. **Softmax**: Normalizes a vector into a probability distribution. Naive implementation
   exp(x) / sum(exp(x)) overflows for large x. Solution: subtract max(x) first.

2. **Log-Sum-Exp (LSE)**: Computes log(sum(exp(x))). Appears in cross-entropy loss,
   attention mechanisms, and probabilistic models. The naive version overflows;
   the stable version uses the identity: LSE(x) = max(x) + log(sum(exp(x - max(x)))).

3. **Log-Softmax**: Combines log and softmax. More stable than log(softmax(x)) because
   it avoids computing softmax explicitly. Uses: log_softmax(x) = x - LSE(x).

4. **Sigmoid**: 1/(1+exp(-x)). For large negative x, exp(-x) overflows. Solution:
   use different formulas for positive vs negative x.

Math
----
# Softmax: softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
# Stable: softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

# Log-Sum-Exp: LSE(x) = log(sum(exp(x)))
# Stable: LSE(x) = max(x) + log(sum(exp(x - max(x))))

# Log-Softmax: log_softmax(x)_i = x_i - LSE(x)

# Sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
# Stable: sigmoid(x) = exp(x) / (1 + exp(x)) if x < 0, else 1 / (1 + exp(-x))

References
----------
- Understanding Deep Learning, Ch. 3: Numerical stability
  https://udlbook.github.io/udlbook/
- Stanford CS231n: Practical tips for training
  https://cs231n.github.io/neural-networks-2/
- NumPy documentation on floating point precision
  https://numpy.org/doc/stable/user/basics.types.html
- "What Every Computer Scientist Should Know About Floating-Point Arithmetic"
  https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html

Implementation Notes
--------------------
- Always use logsumexp instead of log(sum(exp(x)))
- When computing log probabilities, work in log space as long as possible
- For very large inputs (>700 for float64), exp() returns inf
- For very negative inputs (<-745 for float64), exp() returns 0
- Prefer float64 for intermediate computations when possible
"""

# Implementation Status: NOT STARTED
# Complexity: Easy
# Prerequisites: None (foundational module)

import numpy as np
from typing import Union, Optional, Tuple

ArrayLike = Union[np.ndarray, float, list]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute numerically stable softmax.

    Softmax transforms a vector of real numbers into a probability distribution.
    Each output is in (0, 1) and the outputs sum to 1 along the specified axis.

    The stable implementation subtracts the maximum value before exponentiating
    to prevent overflow. This doesn't change the result because:
    exp(x_i - c) / sum(exp(x_j - c)) = exp(x_i) / sum(exp(x_j))

    Args:
        x: Input array of any shape
        axis: Axis along which to compute softmax (default: -1, last axis)

    Returns:
        Array of same shape as x with softmax applied along axis

    Shape:
        Input: (*) where * is any number of dimensions
        Output: Same as input

    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> softmax(x)
        array([0.09003057, 0.24472847, 0.66524096])
        >>> softmax(x).sum()
        1.0

    Implementation Hints:
        1. Find max along axis (keepdims=True for broadcasting)
        2. Subtract max from x
        3. Compute exp of shifted values
        4. Divide by sum along axis
    """
    maximum = x.max(axis=axis, keepdims=True)
    return np.exp(x - maximum) / np.sum(np.exp(x - maximum), axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute numerically stable log-softmax.

    Log-softmax is more numerically stable than computing log(softmax(x))
    separately, especially for extreme values. It's used in cross-entropy
    loss computation.

    The formula log_softmax(x)_i = x_i - logsumexp(x) avoids:
    1. Computing softmax (which can have very small values that underflow in log)
    2. Explicit exponentiation followed by log (loses precision)

    Args:
        x: Input array of any shape
        axis: Axis along which to compute log-softmax

    Returns:
        Array of same shape as x with log-softmax values (all negative)

    Shape:
        Input: (*)
        Output: Same as input

    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> log_softmax(x)
        array([-2.40760596, -1.40760596, -0.40760596])
        >>> np.exp(log_softmax(x))  # Should equal softmax(x)
        array([0.09003057, 0.24472847, 0.66524096])
    """
    return x - logsumexp(x, axis=axis)


def logsumexp(x: np.ndarray, axis: Optional[int] = None,
              keepdims: bool = False) -> np.ndarray:
    """
    Compute log(sum(exp(x))) in a numerically stable way.

    This is one of the most important numerical tricks in ML. The naive
    computation fails because exp(x) overflows for x > ~700 (float64).

    The stable version uses the identity:
    log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))

    Since x - max(x) <= 0, exp(x - max(x)) <= 1, avoiding overflow.
    The exp(0) = 1 term in the sum ensures log argument is always >= 1.

    Args:
        x: Input array
        axis: Axis along which to compute (None = all elements)
        keepdims: Whether to keep reduced dimensions

    Returns:
        Log-sum-exp values

    Shape:
        Input: (*)
        Output: Shape determined by axis and keepdims

    Example:
        >>> x = np.array([1000.0, 1000.0, 1000.0])  # Would overflow naively
        >>> logsumexp(x)  # Stable computation
        1001.0986...  # log(3) + 1000

    Common Uses:
        - Cross-entropy loss: -log(softmax(x)[y]) = -x[y] + logsumexp(x)
        - Partition functions in probabilistic models
        - Marginalizing in log space
    """
    maximum = x.max(axis=axis, keepdims=True)
    return maximum + np.log(np.sum(np.exp(x - maximum), axis=axis, keepdims=True))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute numerically stable sigmoid function.

    Sigmoid squashes real numbers to (0, 1). Used in:
    - Binary classification output layers
    - Gates in LSTMs and GRUs
    - Attention mechanisms (as part of softmax)

    The naive formula 1/(1+exp(-x)) fails for large negative x because
    exp(-x) overflows. Solution: use different formulas for x >= 0 and x < 0.

    For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
    For x < 0:  sigmoid(x) = exp(x) / (1 + exp(x))

    Both are mathematically equivalent but numerically stable in their domains.

    Args:
        x: Input array of any shape

    Returns:
        Sigmoid of x, values in (0, 1)

    Shape:
        Input: (*)
        Output: Same as input

    Example:
        >>> sigmoid(np.array([0.0]))
        array([0.5])
        >>> sigmoid(np.array([-1000.0]))  # Stable, returns ~0
        array([0.])
        >>> sigmoid(np.array([1000.0]))   # Stable, returns ~1
        array([1.])

    Derivative:
        sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        This is useful for backpropagation.
    """
    sigmoid = x.copy()
    sigmoid[x <= 0] = np.exp(x[x <= 0]) / (1 + np.exp(x[x <= 0]))
    sigmoid[x > 0] = 1 / (1 + np.exp(-x[x > 0]))
    return sigmoid


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute numerically stable log(sigmoid(x)).

    Direct computation log(sigmoid(x)) = log(1/(1+exp(-x))) = -log(1+exp(-x))
    is unstable. We use the softplus function: log(sigmoid(x)) = -softplus(-x).

    Args:
        x: Input array

    Returns:
        Log of sigmoid values (all negative)

    Shape:
        Input: (*)
        Output: Same as input
    """
    return -softplus(-x)


def softplus(x: np.ndarray) -> np.ndarray:
    """
    Compute numerically stable softplus: log(1 + exp(x)).

    Softplus is a smooth approximation to ReLU. For large positive x,
    exp(x) overflows. Solution: for large x, softplus(x) ≈ x.

    Args:
        x: Input array

    Returns:
        Softplus of x, always positive

    Shape:
        Input: (*)
        Output: Same as input

    Example:
        >>> softplus(np.array([0.0]))
        array([0.69314718])  # log(2)
        >>> softplus(np.array([100.0]))  # ≈ 100, not inf
        array([100.])
    """
    threshold = 20
    safe = np.minimum(x, threshold)
    x = np.where(x <= threshold, np.log1p(np.exp(safe)), x)
    return x


def clip_gradients(grads: np.ndarray, max_norm: float) -> Tuple[np.ndarray, float]:
    """
    Clip gradients by global norm.

    Gradient clipping prevents exploding gradients during training.
    If ||grads|| > max_norm, scale grads to have norm exactly max_norm.

    Args:
        grads: Gradient array
        max_norm: Maximum allowed norm

    Returns:
        Tuple of (clipped_grads, original_norm)

    Example:
        >>> grads = np.array([3.0, 4.0])  # norm = 5
        >>> clipped, norm = clip_gradients(grads, max_norm=1.0)
        >>> np.linalg.norm(clipped)
        1.0
    """
    return np.clip(grads, min=-max_norm, max=max_norm), np.linalg.norm(grads)


def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class indices to one-hot encoded vectors.

    Args:
        indices: Integer array of class indices, shape (N,) or (N, 1)
        num_classes: Total number of classes

    Returns:
        One-hot encoded array of shape (N, num_classes)

    Example:
        >>> one_hot(np.array([0, 2, 1]), num_classes=3)
        array([[1., 0., 0.],
               [0., 0., 1.],
               [0., 1., 0.]])
    """
    one_hot = np.eye(num_classes)[indices]
    return one_hot


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray,
                         eps: float = 1e-7) -> np.ndarray:
    """
    Compute binary cross-entropy loss element-wise.

    BCE = -[y * log(p) + (1-y) * log(1-p)]

    Args:
        y_pred: Predicted probabilities in (0, 1), shape (N,) or (N, 1)
        y_true: Binary labels {0, 1}, same shape as y_pred
        eps: Small constant for numerical stability

    Returns:
        BCE loss for each sample

    Note:
        Use eps to clip y_pred away from 0 and 1 to avoid log(0).
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """
    L2 normalize array along specified axis.

    Args:
        x: Input array
        axis: Axis along which to normalize
        eps: Small constant to avoid division by zero

    Returns:
        L2-normalized array with unit norm along axis

    Example:
        >>> x = np.array([3.0, 4.0])
        >>> normalize(x)
        array([0.6, 0.8])
    """
    raise NotImplementedError(
        "TODO: Implement L2 normalization\n"
        "Hint: norm = np.linalg.norm(x, axis=axis, keepdims=True)\n"
        "      return x / np.maximum(norm, eps)"
    )


# Aliases for compatibility with test suite
stable_softmax = softmax
stable_sigmoid = sigmoid
stable_logsumexp = logsumexp
