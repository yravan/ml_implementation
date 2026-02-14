"""
Image Transforms
================

Transforms for data augmentation and preprocessing.

Geometry transforms: Resize, Crop, Flip, Rotate, Affine
Color transforms: ColorJitter, Grayscale, Normalize
Composition: Compose, RandomApply, RandomChoice
Auto-augment: AutoAugment, RandAugment, TrivialAugmentWide
Mixing: CutMix, MixUp, RandomErasing

Example:
    >>> from python.vision import transforms
    >>>
    >>> transform = transforms.Compose([
    ...     transforms.RandomResizedCrop(224),
    ...     transforms.RandomHorizontalFlip(),
    ...     transforms.ToTensor(),
    ...     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    ...                          std=[0.229, 0.224, 0.225])
    ... ])
"""

# Geometry transforms
from .geometry import (
    Resize, RandomResize,
    CenterCrop, RandomCrop, RandomResizedCrop, FiveCrop, TenCrop,
    Pad, RandomPad,
    RandomHorizontalFlip, RandomVerticalFlip,
    RandomRotation, RandomAffine, RandomPerspective,
    ElasticTransform,
    ScaleJitter, RandomShortestSize, RandomZoomOut,
)

# Color transforms
from .color import (
    Grayscale, RandomGrayscale, RGB,
    ColorJitter, RandomPhotometricDistort,
    RandomChannelPermutation,
    RandomInvert, RandomPosterize, RandomSolarize,
    RandomAdjustSharpness, RandomAutocontrast, RandomEqualize,
    GaussianBlur, GaussianNoise,
)

# Type conversion
from .type_conversion import (
    ToTensor, ToImage, ToPILImage, ToDtype,
    ConvertImageDtype,
)

# Normalization
from .normalize import Normalize

# Composition
from .container import Compose, RandomApply, RandomChoice, RandomOrder

# Auto-augment
from .auto_augment import (
    AutoAugment, AutoAugmentPolicy,
    RandAugment,
    TrivialAugmentWide,
    AugMix,
)

# Mixing augmentations
from .augment import CutMix, MixUp, RandomErasing

# Misc
from .misc import Identity, Lambda, LinearTransformation

# Functional interface
from . import functional

import numpy as np
from PIL import Image

__all__ = [
    # Geometry
    'Resize', 'RandomResize',
    'CenterCrop', 'RandomCrop', 'RandomResizedCrop', 'FiveCrop', 'TenCrop',
    'Pad', 'RandomPad',
    'RandomHorizontalFlip', 'RandomVerticalFlip',
    'RandomRotation', 'RandomAffine', 'RandomPerspective',
    'ElasticTransform',
    'ScaleJitter', 'RandomShortestSize', 'RandomZoomOut',
    # Color
    'Grayscale', 'RandomGrayscale', 'RGB',
    'ColorJitter', 'RandomPhotometricDistort',
    'RandomChannelPermutation',
    'RandomInvert', 'RandomPosterize', 'RandomSolarize',
    'RandomAdjustSharpness', 'RandomAutocontrast', 'RandomEqualize',
    'GaussianBlur', 'GaussianNoise',
    # Type conversion
    'ToTensor', 'ToImage', 'ToPILImage', 'ToDtype', 'ConvertImageDtype',
    # Normalization
    'Normalize',
    # Composition
    'Compose', 'RandomApply', 'RandomChoice', 'RandomOrder',
    # Auto-augment
    'AutoAugment', 'AutoAugmentPolicy', 'RandAugment', 'TrivialAugmentWide', 'AugMix',
    # Mixing
    'CutMix', 'MixUp', 'RandomErasing',
    # Misc
    'Identity', 'Lambda', 'LinearTransformation',
    # Functional
    'functional',
]

def load_image(path: str, size: int) -> np.ndarray:
    """
    Load and preprocess a single image.

    Args:
        path: Path to image file
        size: Target size (square crop)

    Returns:
        numpy array of shape (3, size, size), float32, 0-1 range
    """
    # Convert to numpy: (H, W, 3) uint8 0-255
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    return arr

