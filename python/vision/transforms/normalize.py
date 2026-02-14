"""
Normalization Transform
=======================
"""

import numpy as np
from typing import Sequence


class Normalize:
    """
    Normalize image with mean and standard deviation.

    output = (input - mean) / std

    Common values:
        ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    Args:
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel
        inplace: Whether to modify in place
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        inplace: bool = False,
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.inplace = inplace

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.inplace:
            img = img.astype(np.float32)
            img -= self.mean[:, None, None]
            img /= self.std[:, None, None]
            return img
        else:
            return (img - self.mean[:, None, None]) / self.std[:, None, None]

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean.tolist()}, std={self.std.tolist()})'
