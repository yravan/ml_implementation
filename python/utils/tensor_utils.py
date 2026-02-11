"""
Tensor Utilities
================

Helper functions for tensor manipulation, shape checking, and broadcasting.

Theory
------
Working with multi-dimensional arrays (tensors) is fundamental to deep learning.
Common operations include reshaping, broadcasting, and shape validation. These
utilities help catch bugs early and make code more readable.

NumPy's broadcasting rules allow operations between arrays of different shapes:
1. If arrays have different ndim, prepend 1s to the smaller array's shape
2. Arrays with size 1 in a dimension are stretched to match the other array
3. Arrays must have the same size or one must have size 1 in each dimension

Understanding shapes is critical for debugging neural networks. Most bugs in
ML code are shape mismatches that could be caught with proper assertions.

Math
----
# Broadcasting example:
# A: (3, 4, 1) + B: (1, 5) -> Result: (3, 4, 5)
#
# Step 1: Align shapes from right
#   A: (3, 4, 1)
#   B:    (1, 5)
# Step 2: Prepend 1s to B -> (1, 1, 5)
# Step 3: Broadcast: (3, 4, 1) + (1, 1, 5) -> (3, 4, 5)

# Common shape transformations:
# Flatten: (B, C, H, W) -> (B, C*H*W)
# Reshape for attention: (B, T, D) -> (B, T, H, D/H) -> (B, H, T, D/H)

References
----------
- NumPy Broadcasting documentation
  https://numpy.org/doc/stable/user/basics.broadcasting.html
- CS231n: Numpy Tutorial
  https://cs231n.github.io/python-numpy-tutorial/
- "Einsum is All You Need" (Einstein summation)
  https://rockt.github.io/2018/04/30/einsum

Implementation Notes
--------------------
- Always validate shapes at layer boundaries during development
- Use descriptive variable names for dimensions (batch, seq_len, hidden_dim)
- Prefer reshape over flatten for reversibility
- einsum is powerful but can be hard to read - use comments
"""

# Implementation Status: NOT STARTED
# Complexity: Easy
# Prerequisites: None (foundational module)

import numpy as np
from typing import Tuple, List, Optional, Union


def check_shape(arr: np.ndarray, expected: Tuple[Optional[int], ...],
                name: str = "array") -> None:
    """
    Assert that array has expected shape, allowing None for any dimension.

    This is a debugging utility to catch shape mismatches early. Use it
    liberally during development and at layer boundaries.

    Args:
        arr: Array to check
        expected: Expected shape tuple. Use None for dimensions that can vary.
        name: Name of array for error message

    Raises:
        AssertionError: If shape doesn't match

    Example:
        >>> x = np.zeros((32, 10, 64))
        >>> check_shape(x, (32, None, 64), "hidden")  # Passes
        >>> check_shape(x, (32, 10, 128), "hidden")   # Raises AssertionError

    Usage Pattern:
        def forward(self, x):
            check_shape(x, (None, self.in_features), "input")
            out = x @ self.weight + self.bias
            check_shape(out, (None, self.out_features), "output")
            return out
    """
    assert arr.ndim == len(expected), f'{name} wrong ndim'
    for actual, exp in zip(arr.shape, expected):
        if exp is not None:
            assert actual == exp, f'{name} shape mismatch'


def get_fans(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Compute fan_in and fan_out for a weight tensor.

    Fan-in is the number of input units, fan-out is the number of output units.
    These are used in weight initialization (Xavier, Kaiming).

    For different layer types:
    - Linear (in, out): fan_in=in, fan_out=out
    - Conv2D (out, in, kH, kW): fan_in=in*kH*kW, fan_out=out*kH*kW
    - Conv1D (out, in, k): fan_in=in*k, fan_out=out*k

    Args:
        shape: Shape of weight tensor

    Returns:
        Tuple of (fan_in, fan_out)

    Example:
        >>> get_fans((784, 256))  # Linear layer
        (784, 256)
        >>> get_fans((64, 32, 3, 3))  # Conv2D layer
        (288, 576)  # 32*3*3, 64*3*3
    """
    if len(shape) == 2:
        return shape[0], shape[1]
    elif len(shape) >= 3:
        receptive_field = np.prod(shape[2:])
        fan_in = shape[0] * receptive_field
        fan_out = shape[1] * receptive_field
        return fan_in, fan_out
    raise NotImplementedError(
        "TODO: Implement fan computation\n"
        "Hint:\n"
        "  if len(shape) == 2:  # Linear\n"
        "      return shape[0], shape[1]\n"
        "  elif len(shape) >= 3:  # Conv\n"
        "      receptive_field = np.prod(shape[2:])\n"
        "      fan_in = shape[1] * receptive_field\n"
        "      fan_out = shape[0] * receptive_field\n"
        "      return fan_in, fan_out"
    )


def flatten(x: np.ndarray, start_dim: int = 1, end_dim: int = -1) -> np.ndarray:
    """
    Flatten dimensions from start_dim to end_dim (inclusive).

    Common use cases:
    - Flatten spatial dims in CNN: (B, C, H, W) -> (B, C*H*W) with start_dim=1
    - Flatten for batched matmul: (B, H, T, D) -> (B*H, T, D) with start_dim=0, end_dim=1

    Args:
        x: Input array
        start_dim: First dimension to flatten (default: 1, keeps batch)
        end_dim: Last dimension to flatten (default: -1, last)

    Returns:
        Flattened array

    Example:
        >>> x = np.zeros((2, 3, 4, 5))
        >>> flatten(x).shape
        (2, 60)  # 3*4*5
        >>> flatten(x, start_dim=0, end_dim=1).shape
        (6, 4, 5)  # 2*3
    """
    start_shape = x.shape
    new_shape = (*start_shape[:start_dim], -1, *start_shape[end_dim + 1:])
    x = np.reshape(x, new_shape)
    return x
    raise NotImplementedError(
        "TODO: Implement flexible flatten\n"
        "Hint:\n"
        "  if end_dim < 0:\n"
        "      end_dim = x.ndim + end_dim\n"
        "  new_shape = x.shape[:start_dim] + (-1,) + x.shape[end_dim+1:]\n"
        "  return x.reshape(new_shape)"
    )


def unflatten(x: np.ndarray, dim: int, sizes: Tuple[int, ...]) -> np.ndarray:
    """
    Expand a flattened dimension back to original shape.

    Inverse of flatten for a single dimension.

    Args:
        x: Input array
        dim: Dimension to unflatten
        sizes: Shape to expand to (must match dimension size)

    Returns:
        Unflattened array

    Example:
        >>> x = np.zeros((2, 60))  # Flattened (2, 3, 4, 5)
        >>> unflatten(x, dim=1, sizes=(3, 4, 5)).shape
        (2, 3, 4, 5)
    """
    raise NotImplementedError(
        "TODO: Implement unflatten\n"
        "Hint:\n"
        "  assert x.shape[dim] == np.prod(sizes)\n"
        "  new_shape = x.shape[:dim] + sizes + x.shape[dim+1:]\n"
        "  return x.reshape(new_shape)"
    )


def repeat_interleave(x: np.ndarray, repeats: int, axis: int) -> np.ndarray:
    """
    Repeat elements of array along axis.

    Different from np.tile: repeat_interleave repeats each element,
    tile repeats the whole array.

    Args:
        x: Input array
        repeats: Number of times to repeat
        axis: Axis along which to repeat

    Returns:
        Array with elements repeated

    Example:
        >>> x = np.array([[1, 2], [3, 4]])
        >>> repeat_interleave(x, 2, axis=0)
        array([[1, 2], [1, 2], [3, 4], [3, 4]])
        >>> repeat_interleave(x, 2, axis=1)
        array([[1, 1, 2, 2], [3, 3, 4, 4]])

    Use Case:
        In multi-head attention, repeat key/value for grouped-query attention.
    """
    raise NotImplementedError(
        "TODO: Implement repeat_interleave\n"
        "Hint: return np.repeat(x, repeats, axis=axis)"
    )


def batched_index_select(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Select elements from batched tensor using indices.

    For each batch, select elements at given indices. Useful for
    embedding lookup, gathering predictions at specific positions.

    Args:
        x: Input array of shape (batch, seq_len, features)
        indices: Index array of shape (batch,) or (batch, num_indices)

    Returns:
        Selected elements

    Example:
        >>> x = np.arange(24).reshape(2, 4, 3)  # (batch=2, seq=4, feat=3)
        >>> indices = np.array([1, 2])  # Select position 1 from batch 0, position 2 from batch 1
        >>> batched_index_select(x, indices)
        array([[ 3,  4,  5],
               [18, 19, 20]])  # x[0, 1, :] and x[1, 2, :]
    """
    return np.take(x, indices, axis=-1)


def pad_sequence(sequences: List[np.ndarray], padding_value: float = 0.0,
                 max_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad a list of variable-length sequences to the same length.

    Args:
        sequences: List of arrays, each of shape (seq_len_i, features)
        padding_value: Value to use for padding
        max_len: Maximum length (None = max of all sequences)

    Returns:
        Tuple of:
        - Padded array of shape (batch, max_len, features)
        - Lengths array of shape (batch,)

    Example:
        >>> seqs = [np.ones((3, 4)), np.ones((5, 4)), np.ones((2, 4))]
        >>> padded, lengths = pad_sequence(seqs)
        >>> padded.shape
        (3, 5, 4)
        >>> lengths
        array([3, 5, 2])
    """
    raise NotImplementedError(
        "TODO: Implement sequence padding\n"
        "Hint:\n"
        "  lengths = np.array([len(s) for s in sequences])\n"
        "  max_len = max_len or max(lengths)\n"
        "  batch_size = len(sequences)\n"
        "  features = sequences[0].shape[1] if sequences[0].ndim > 1 else 1\n"
        "  padded = np.full((batch_size, max_len, features), padding_value)\n"
        "  for i, seq in enumerate(sequences):\n"
        "      padded[i, :len(seq)] = seq"
    )


def create_causal_mask(seq_len: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Create a causal (lower triangular) attention mask.

    In autoregressive models, each position can only attend to
    previous positions (including itself). This mask has True/1
    where attention is allowed.

    Args:
        seq_len: Sequence length
        dtype: Output dtype

    Returns:
        Mask of shape (seq_len, seq_len)

    Example:
        >>> create_causal_mask(4)
        array([[1., 0., 0., 0.],
               [1., 1., 0., 0.],
               [1., 1., 1., 0.],
               [1., 1., 1., 1.]])

    Usage:
        In attention: scores = scores.masked_fill(mask == 0, -inf)
    """
    return np.tril(np.ones((seq_len, seq_len)), -1, dtype=dtype)


def create_padding_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    """
    Create a padding mask from sequence lengths.

    Args:
        lengths: Array of actual sequence lengths, shape (batch,)
        max_len: Maximum sequence length (padded length)

    Returns:
        Boolean mask of shape (batch, max_len) where True = valid token

    Example:
        >>> create_padding_mask(np.array([2, 3, 1]), max_len=4)
        array([[ True,  True, False, False],
               [ True,  True,  True, False],
               [ True, False, False, False]])
    """
    mask = np.ones((max_len, max_len), dtype=bool)
    mask[:, :lengths] = False
    return mask


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Split last dimension for multi-head attention.

    Reshapes (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)
    where head_dim = d_model / num_heads.

    Args:
        x: Input of shape (batch, seq_len, d_model)
        num_heads: Number of attention heads

    Returns:
        Reshaped array (batch, num_heads, seq_len, head_dim)

    Example:
        >>> x = np.zeros((2, 10, 64))
        >>> split_heads(x, num_heads=8).shape
        (2, 8, 10, 8)  # head_dim = 64/8 = 8
    """
    batch, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    new_shape = (batch, seq_len, num_heads, head_dim)
    x = np.reshape(x, new_shape)
    x = np.transpose(x, (0, 2, 1, 3))
    return x


def merge_heads(x: np.ndarray) -> np.ndarray:
    """
    Merge multi-head attention output back to original shape.

    Inverse of split_heads:
    (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)

    Args:
        x: Input of shape (batch, num_heads, seq_len, head_dim)

    Returns:
        Merged array (batch, seq_len, d_model)

    Example:
        >>> x = np.zeros((2, 8, 10, 8))
        >>> merge_heads(x).shape
        (2, 10, 64)
    """
    x = x.transpose((0, 2, 1, 3))
    batch, seq_len, num_heads, head_dim = x.shape
    new_shape = (batch, seq_len, num_heads * head_dim)
    x = np.reshape(x, new_shape)
    return x


def broadcast_shapes(*shapes) -> tuple:
    """
    Compute the broadcast shape of multiple array shapes.

    Args:
        *shapes: Variable number of shape tuples

    Returns:
        Resulting broadcast shape

    Example:
        >>> broadcast_shapes((3, 1), (1, 4))
        (3, 4)
    """
    raise NotImplementedError(
        "TODO: Implement broadcast shape computation\n"
        "Hint: Use np.broadcast_shapes or implement manually"
    )
