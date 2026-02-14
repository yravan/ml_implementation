"""
Functional Interface for Transforms
===================================

Stateless functions for image transformations.
All functions operate on numpy arrays.
"""

import numpy as np
from typing import Tuple, Union, Optional, List


# =============================================================================
# Geometry Functions
# =============================================================================

def resize(
    img: np.ndarray,
    size: Union[int, Tuple[int, int]],
    interpolation: str = 'bilinear',
    antialias: bool = True,
) -> np.ndarray:
    """Resize image."""
    raise NotImplementedError("TODO: Implement resize")


def center_crop(img: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Center crop image."""
    raise NotImplementedError("TODO: Implement center_crop")


def crop(img: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    """Crop image at specified location."""
    raise NotImplementedError("TODO: Implement crop")


def pad(
    img: np.ndarray,
    padding: Union[int, Tuple[int, ...]],
    fill: Union[int, Tuple[int, ...]] = 0,
    padding_mode: str = 'constant',
) -> np.ndarray:
    """Pad image."""
    raise NotImplementedError("TODO: Implement pad")


def hflip(img: np.ndarray) -> np.ndarray:
    """Horizontally flip image."""
    raise NotImplementedError("TODO: Implement hflip")


def vflip(img: np.ndarray) -> np.ndarray:
    """Vertically flip image."""
    raise NotImplementedError("TODO: Implement vflip")


def rotate(
    img: np.ndarray,
    angle: float,
    interpolation: str = 'nearest',
    expand: bool = False,
    center: Optional[Tuple[int, int]] = None,
    fill: Union[int, Tuple[int, ...]] = 0,
) -> np.ndarray:
    """Rotate image by angle."""
    raise NotImplementedError("TODO: Implement rotate")


def affine(
    img: np.ndarray,
    angle: float,
    translate: Tuple[float, float],
    scale: float,
    shear: Tuple[float, float],
    interpolation: str = 'nearest',
    fill: Union[int, Tuple[int, ...]] = 0,
) -> np.ndarray:
    """Apply affine transformation."""
    raise NotImplementedError("TODO: Implement affine")


def perspective(
    img: np.ndarray,
    startpoints: List[Tuple[int, int]],
    endpoints: List[Tuple[int, int]],
    interpolation: str = 'bilinear',
    fill: Union[int, Tuple[int, ...]] = 0,
) -> np.ndarray:
    """Apply perspective transformation."""
    raise NotImplementedError("TODO: Implement perspective")


def elastic_transform(
    img: np.ndarray,
    displacement: np.ndarray,
    interpolation: str = 'bilinear',
    fill: Union[int, Tuple[int, ...]] = 0,
) -> np.ndarray:
    """Apply elastic transformation."""
    raise NotImplementedError("TODO: Implement elastic_transform")


# =============================================================================
# Color Functions
# =============================================================================

def rgb_to_grayscale(img: np.ndarray, num_output_channels: int = 1) -> np.ndarray:
    """Convert RGB to grayscale."""
    raise NotImplementedError("TODO: Implement rgb_to_grayscale")


def adjust_brightness(img: np.ndarray, brightness_factor: float) -> np.ndarray:
    """Adjust brightness."""
    raise NotImplementedError("TODO: Implement adjust_brightness")


def adjust_contrast(img: np.ndarray, contrast_factor: float) -> np.ndarray:
    """Adjust contrast."""
    raise NotImplementedError("TODO: Implement adjust_contrast")


def adjust_saturation(img: np.ndarray, saturation_factor: float) -> np.ndarray:
    """Adjust saturation."""
    raise NotImplementedError("TODO: Implement adjust_saturation")


def adjust_hue(img: np.ndarray, hue_factor: float) -> np.ndarray:
    """Adjust hue."""
    raise NotImplementedError("TODO: Implement adjust_hue")


def adjust_gamma(img: np.ndarray, gamma: float, gain: float = 1.0) -> np.ndarray:
    """Adjust gamma."""
    raise NotImplementedError("TODO: Implement adjust_gamma")


def adjust_sharpness(img: np.ndarray, sharpness_factor: float) -> np.ndarray:
    """Adjust sharpness."""
    raise NotImplementedError("TODO: Implement adjust_sharpness")


def invert(img: np.ndarray) -> np.ndarray:
    """Invert colors."""
    raise NotImplementedError("TODO: Implement invert")


def posterize(img: np.ndarray, bits: int) -> np.ndarray:
    """Posterize image."""
    raise NotImplementedError("TODO: Implement posterize")


def solarize(img: np.ndarray, threshold: float) -> np.ndarray:
    """Solarize image."""
    raise NotImplementedError("TODO: Implement solarize")


def autocontrast(img: np.ndarray) -> np.ndarray:
    """Apply autocontrast."""
    raise NotImplementedError("TODO: Implement autocontrast")


def equalize(img: np.ndarray) -> np.ndarray:
    """Equalize histogram."""
    raise NotImplementedError("TODO: Implement equalize")


def gaussian_blur(
    img: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]],
    sigma: Tuple[float, float],
) -> np.ndarray:
    """Apply Gaussian blur."""
    raise NotImplementedError("TODO: Implement gaussian_blur")


# =============================================================================
# Normalization Functions
# =============================================================================

def normalize(
    tensor: np.ndarray,
    mean: List[float],
    std: List[float],
    inplace: bool = False,
) -> np.ndarray:
    """Normalize tensor with mean and std."""
    raise NotImplementedError("TODO: Implement normalize")


# =============================================================================
# Type Conversion Functions
# =============================================================================

def to_tensor(pic: np.ndarray) -> np.ndarray:
    """Convert image to tensor format (CHW float [0, 1])."""
    raise NotImplementedError("TODO: Implement to_tensor")


def to_image(tensor: np.ndarray) -> np.ndarray:
    """Convert tensor to image format (HWC uint8 [0, 255])."""
    raise NotImplementedError("TODO: Implement to_image")


# =============================================================================
# Augmentation Functions
# =============================================================================

def erase(
    img: np.ndarray,
    i: int,
    j: int,
    h: int,
    w: int,
    v: Union[float, np.ndarray],
    inplace: bool = False,
) -> np.ndarray:
    """Erase rectangular region."""
    raise NotImplementedError("TODO: Implement erase")


def cutmix(
    img1: np.ndarray,
    img2: np.ndarray,
    lam: float,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """Apply CutMix to two images."""
    raise NotImplementedError("TODO: Implement cutmix")


def mixup(
    img1: np.ndarray,
    img2: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Apply MixUp to two images."""
    raise NotImplementedError("TODO: Implement mixup")
