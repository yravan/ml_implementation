"""
GoogLeNet (Inception v1)
========================

GoogLeNet model architecture from "Going Deeper with Convolutions"
https://arxiv.org/abs/1409.4842

Key innovation: Inception module - parallel branches with different receptive fields
    - 1x1 conv (captures fine details)
    - 1x1 conv -> 3x3 conv (medium receptive field)
    - 1x1 conv -> 5x5 conv (large receptive field)
    - 3x3 maxpool -> 1x1 conv (preserves spatial info)
    - All branches concatenated

Design philosophy: Let the network learn which filter size is optimal at each layer
by providing all options simultaneously.

Also introduces auxiliary classifiers during training to combat vanishing gradients.
"""

from typing import Optional, Tuple
from python.nn_core import Module


class Inception(Module):
    """
    Inception module with dimension reductions.

    Args:
        in_channels: Input channels
        ch1x1: Output channels for 1x1 conv branch
        ch3x3red: Reduction channels before 3x3 conv
        ch3x3: Output channels for 3x3 conv branch
        ch5x5red: Reduction channels before 5x5 conv
        ch5x5: Output channels for 5x5 conv branch
        pool_proj: Output channels for pooling branch
    """

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
    ):
        super().__init__()
        # TODO: Implement Inception module
        # Branch 1: 1x1 conv
        # Branch 2: 1x1 conv -> 3x3 conv
        # Branch 3: 1x1 conv -> 5x5 conv
        # Branch 4: 3x3 maxpool -> 1x1 conv
        # Output: Concatenate all branches
        raise NotImplementedError("TODO: Implement Inception module")

    def forward(self, x):
        """Forward pass through all branches, concatenate outputs."""
        raise NotImplementedError("TODO: Implement forward")


class InceptionAux(Module):
    """
    Auxiliary classifier for training.

    Args:
        in_channels: Input channels
        num_classes: Number of output classes
        dropout: Dropout probability
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.7):
        super().__init__()
        # TODO: Implement auxiliary classifier
        # AvgPool2d(5, stride=3)
        # Conv2d(in_channels, 128, 1)
        # Flatten
        # Linear(128 * 4 * 4, 1024)
        # Dropout(dropout)
        # Linear(1024, num_classes)
        raise NotImplementedError("TODO: Implement InceptionAux")


class GoogLeNet(Module):
    """
    GoogLeNet (Inception v1) model.

    Args:
        num_classes: Number of output classes
        aux_logits: Whether to include auxiliary classifiers
        dropout: Dropout probability
        dropout_aux: Dropout for auxiliary classifiers
    """

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
    ):
        super().__init__()
        self.aux_logits = aux_logits

        # TODO: Implement GoogLeNet
        # See original paper for full architecture
        raise NotImplementedError("TODO: Implement GoogLeNet")

    def forward(self, x):
        """
        Forward pass.

        Returns:
            If training with aux_logits: (main_output, aux1_output, aux2_output)
            Otherwise: main_output
        """
        raise NotImplementedError("TODO: Implement forward")


def googlenet(num_classes: int = 1000, **kwargs) -> GoogLeNet:
    """GoogLeNet (Inception v1) model."""
    return GoogLeNet(num_classes=num_classes, **kwargs)
