"""
Mixing Augmentations
====================

Augmentations that mix multiple images or mask regions.
"""

import numpy as np
from typing import Tuple, Union


class CutMix:
    """
    CutMix augmentation from "CutMix: Regularization Strategy to Train Strong Classifiers"
    https://arxiv.org/abs/1905.04899

    Cuts and pastes patches between training images.
    Labels are mixed proportionally to the area of patches.

    Args:
        alpha: Parameter for Beta distribution (controls patch size)

    Note: This is typically applied at the batch level during training.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self,
        batch_images: np.ndarray,
        batch_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CutMix to a batch.

        Args:
            batch_images: (N, C, H, W) image batch
            batch_labels: (N,) or (N, num_classes) labels

        Returns:
            Mixed images and mixed labels
        """
        raise NotImplementedError("TODO: Implement CutMix")


class MixUp:
    """
    MixUp augmentation from "mixup: Beyond Empirical Risk Minimization"
    https://arxiv.org/abs/1710.09412

    Linearly interpolates between pairs of images and labels.

    Args:
        alpha: Parameter for Beta distribution

    Note: This is typically applied at the batch level during training.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        batch_images: np.ndarray,
        batch_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply MixUp to a batch.

        Args:
            batch_images: (N, C, H, W) image batch
            batch_labels: (N,) or (N, num_classes) labels

        Returns:
            Mixed images and mixed labels
        """
        raise NotImplementedError("TODO: Implement MixUp")


class RandomErasing:
    """
    Random Erasing augmentation from "Random Erasing Data Augmentation"
    https://arxiv.org/abs/1708.04896

    Randomly erases a rectangular region (similar to Cutout).

    Args:
        p: Probability of erasing
        scale: Range of proportion of area to erase
        ratio: Range of aspect ratio of erased area
        value: Erasing value (number, 'random', or tuple of channel values)
        inplace: Whether to modify in place
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Union[float, str, Tuple[float, ...]] = 0,
        inplace: bool = False,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomErasing")
