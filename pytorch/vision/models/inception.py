"""
Inception v3
============

Inception v3 model architecture from "Rethinking the Inception Architecture for Computer Vision"
https://arxiv.org/abs/1512.00567

Improvements over GoogLeNet:
    1. Factorized convolutions (7x7 -> 1x7 + 7x1)
    2. Label smoothing regularization
    3. Batch normalization
    4. RMSProp optimizer

Key design principles:
    - Avoid representational bottlenecks (don't reduce too aggressively)
    - Higher dimensional representations are easier to process
    - Spatial aggregation can be done over lower dimensional embeddings
    - Balance network width and depth

Input size: 299x299 (different from other models!)
"""

from typing import Optional, Tuple, List
from python.nn_core import Module


class InceptionA(Module):
    """Inception module with 1x1, 5x5, 3x3 double, and pool branches."""

    def __init__(self, in_channels: int, pool_features: int):
        super().__init__()
        # TODO: Implement InceptionA
        raise NotImplementedError("TODO: Implement InceptionA")


class InceptionB(Module):
    """Inception module for grid reduction."""

    def __init__(self, in_channels: int):
        super().__init__()
        # TODO: Implement InceptionB
        raise NotImplementedError("TODO: Implement InceptionB")


class InceptionC(Module):
    """Inception module with factorized 7x7 convolutions."""

    def __init__(self, in_channels: int, channels_7x7: int):
        super().__init__()
        # TODO: Implement InceptionC (uses 1x7 and 7x1 convs)
        raise NotImplementedError("TODO: Implement InceptionC")


class InceptionD(Module):
    """Inception module for grid reduction."""

    def __init__(self, in_channels: int):
        super().__init__()
        # TODO: Implement InceptionD
        raise NotImplementedError("TODO: Implement InceptionD")


class InceptionE(Module):
    """Inception module with expanded filter bank outputs."""

    def __init__(self, in_channels: int):
        super().__init__()
        # TODO: Implement InceptionE
        raise NotImplementedError("TODO: Implement InceptionE")


class InceptionAux(Module):
    """Auxiliary classifier for Inception v3."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # TODO: Implement auxiliary classifier
        raise NotImplementedError("TODO: Implement InceptionAux")


class InceptionV3(Module):
    """
    Inception v3 model.

    Args:
        num_classes: Number of output classes
        aux_logits: Whether to use auxiliary classifier
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.aux_logits = aux_logits

        # TODO: Implement Inception v3
        # Input: 299x299x3
        # Initial convolutions
        # Mixed_5b, 5c, 5d (InceptionA)
        # Mixed_6a (InceptionB - reduction)
        # Mixed_6b, 6c, 6d, 6e (InceptionC)
        # Mixed_7a (InceptionD - reduction)
        # Mixed_7b, 7c (InceptionE)
        # Pooling + classifier
        raise NotImplementedError("TODO: Implement InceptionV3")

    def forward(self, x):
        """
        Forward pass.
        Note: Expects 299x299 input!
        """
        raise NotImplementedError("TODO: Implement forward")


def inception_v3(num_classes: int = 1000, **kwargs) -> InceptionV3:
    """Inception v3 model."""
    return InceptionV3(num_classes=num_classes, **kwargs)
