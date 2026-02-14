"""
Auto-Augmentation Transforms
============================

Learned or hand-designed augmentation policies.
"""

import numpy as np
from typing import List, Tuple, Optional
from enum import Enum


class AutoAugmentPolicy(Enum):
    """Available AutoAugment policies."""
    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    SVHN = "svhn"


class AutoAugment:
    """
    AutoAugment as described in "AutoAugment: Learning Augmentation Policies from Data"
    https://arxiv.org/abs/1805.09501

    Applies learned augmentation policies.

    Args:
        policy: Which policy to use (IMAGENET, CIFAR10, SVHN)

    The policy defines pairs of operations, each applied with probability and magnitude.
    """

    def __init__(self, policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET):
        self.policy = policy
        self.transforms = self._get_transforms(policy)

    def _get_transforms(self, policy: AutoAugmentPolicy) -> List:
        """Get transforms for the given policy."""
        raise NotImplementedError("TODO: Implement policy transforms")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement AutoAugment")


class RandAugment:
    """
    RandAugment as described in "RandAugment: Practical automated data augmentation"
    https://arxiv.org/abs/1909.13719

    Simpler than AutoAugment: randomly select N ops, each with magnitude M.

    Args:
        num_ops: Number of augmentation operations to apply
        magnitude: Magnitude for all operations (0-30)
        num_magnitude_bins: Number of magnitude levels
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
    ):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandAugment")


class TrivialAugmentWide:
    """
    TrivialAugment as described in "TrivialAugment: Tuning-free Yet State-of-the-Art"
    https://arxiv.org/abs/2103.10158

    Even simpler: randomly select ONE operation with random magnitude.

    Args:
        num_magnitude_bins: Number of magnitude levels
    """

    def __init__(self, num_magnitude_bins: int = 31):
        self.num_magnitude_bins = num_magnitude_bins

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement TrivialAugmentWide")


class AugMix:
    """
    AugMix as described in "AugMix: A Simple Data Processing Method to Improve Robustness"
    https://arxiv.org/abs/1912.02781

    Mixes multiple augmented images for improved robustness.

    Args:
        severity: Severity of augmentations (1-10)
        mixture_width: Number of augmentation chains to mix
        chain_depth: Depth of each augmentation chain (-1 for random)
        alpha: Dirichlet concentration parameter
    """

    def __init__(
        self,
        severity: int = 3,
        mixture_width: int = 3,
        chain_depth: int = -1,
        alpha: float = 1.0,
    ):
        self.severity = severity
        self.mixture_width = mixture_width
        self.chain_depth = chain_depth
        self.alpha = alpha

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement AugMix")
