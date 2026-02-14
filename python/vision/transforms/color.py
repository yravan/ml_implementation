"""
Color Transforms
================

Color and pixel-level transformations.
"""

import numpy as np
from typing import Tuple, Union, Optional, Sequence


class Grayscale:
    """
    Convert image to grayscale.

    Args:
        num_output_channels: 1 or 3 (grayscale or 3-channel grayscale)
    """

    def __init__(self, num_output_channels: int = 1):
        self.num_output_channels = num_output_channels

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement Grayscale")


class RandomGrayscale:
    """Randomly convert to grayscale."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomGrayscale")


class RGB:
    """Convert image to RGB."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RGB")


class ColorJitter:
    """
    Randomly change brightness, contrast, saturation, and hue.

    Args:
        brightness: Brightness jitter factor (0 = no change)
        contrast: Contrast jitter factor
        saturation: Saturation jitter factor
        hue: Hue jitter factor (in [-0.5, 0.5])
    """

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
    ):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5))

    def _check_input(self, value, name, center=1, bound=(0, float('inf'))):
        """Convert input to (min, max) range."""
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            return (max(center - value, bound[0]), center + value)
        return value

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement ColorJitter")


class RandomPhotometricDistort:
    """
    Random photometric distortions as used in SSD.
    Applies brightness, contrast, saturation, hue in random order.
    """

    def __init__(
        self,
        brightness: Tuple[float, float] = (0.875, 1.125),
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.5, 1.5),
        hue: Tuple[float, float] = (-0.05, 0.05),
        p: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomPhotometricDistort")


class RandomChannelPermutation:
    """Randomly permute color channels."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomChannelPermutation")


class RandomInvert:
    """Randomly invert colors."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomInvert")


class RandomPosterize:
    """Reduce number of bits for each color channel."""

    def __init__(self, bits: int, p: float = 0.5):
        self.bits = bits
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomPosterize")


class RandomSolarize:
    """Invert pixels above threshold."""

    def __init__(self, threshold: float, p: float = 0.5):
        self.threshold = threshold
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomSolarize")


class RandomAdjustSharpness:
    """Randomly adjust sharpness."""

    def __init__(self, sharpness_factor: float, p: float = 0.5):
        self.sharpness_factor = sharpness_factor
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomAdjustSharpness")


class RandomAutocontrast:
    """Randomly apply autocontrast."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomAutocontrast")


class RandomEqualize:
    """Randomly equalize histogram."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomEqualize")


class GaussianBlur:
    """Apply Gaussian blur."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        sigma: Tuple[float, float] = (0.1, 2.0),
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement GaussianBlur")


class GaussianNoise:
    """Add Gaussian noise."""

    def __init__(self, mean: float = 0.0, sigma: float = 0.1):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement GaussianNoise")
