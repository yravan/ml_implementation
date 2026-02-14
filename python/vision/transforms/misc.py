"""
Miscellaneous Transforms
========================
"""

import numpy as np
from typing import Callable


class Identity:
    """Identity transform (returns input unchanged)."""

    def __call__(self, x):
        return x


class Lambda:
    """
    Apply a user-defined lambda function.

    Args:
        lambd: Lambda function to apply

    Example:
        >>> transform = Lambda(lambda x: x * 2)
    """

    def __init__(self, lambd: Callable):
        self.lambd = lambd

    def __call__(self, x):
        return self.lambd(x)


class LinearTransformation:
    """
    Apply a linear transformation (whitening).

    transform = (x - mean) @ transformation_matrix

    Args:
        transformation_matrix: (D, D) transformation matrix
        mean_vector: (D,) mean vector
    """

    def __init__(self, transformation_matrix: np.ndarray, mean_vector: np.ndarray):
        self.transformation_matrix = transformation_matrix
        self.mean_vector = mean_vector

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement LinearTransformation")
