"""
Type Conversion Transforms
==========================

Convert between different image representations.
"""

import numpy as np
from typing import Union


class ToTensor:
    """
    Convert image to tensor format.

    Converts HCW uint8 [0, 255] to CHW float [0, 1].
    """

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert img.dtype == np.uint8
        assert img.ndim == 3
        img = ((255 - img) / 255.0).astype(np.float32)
        img = img.transpose(2, 0, 1)
        return img


class ToImage:
    """
    Convert tensor to image format.

    Converts CHW float [0, 1] to HWC uint8 [0, 255].
    """

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        assert tensor.dtype == np.float32
        assert tensor.ndim == 3
        tensor = tensor.transpose(2, 0, 1) * 255
        tensor = tensor.astype(np.uint8)
        return tensor

class ToPILImage:
    """
    Convert tensor to PIL Image.

    Note: Requires PIL library.
    """

    def __init__(self, mode: str = None):
        self.mode = mode

    def __call__(self, tensor: np.ndarray):
        raise NotImplementedError("TODO: Implement ToPILImage")


class ToDtype:
    """
    Convert image dtype.

    Args:
        dtype: Target numpy dtype
        scale: Whether to scale values
    """

    def __init__(self, dtype: np.dtype, scale: bool = False):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement ToDtype")


class ConvertImageDtype:
    """
    Convert image to specific dtype with proper scaling.

    Handles scaling between uint8 [0, 255] and float [0, 1].
    """

    def __init__(self, dtype: np.dtype):
        self.dtype = dtype

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement ConvertImageDtype")
