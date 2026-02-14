"""
MNASNet
=======

MNASNet architecture from "MnasNet: Platform-Aware Neural Architecture Search for Mobile"
https://arxiv.org/abs/1807.11626

Key innovation: Multi-objective Neural Architecture Search
    - Optimizes for both accuracy AND latency
    - Uses reinforcement learning with factorized hierarchical search space
    - Platform-aware: directly measures latency on target device

Architecture features:
    - Similar to MobileNetV2 inverted residuals
    - Squeeze-and-Excitation blocks
    - Lightweight attention
"""

from typing import List
from python.nn_core import Module


class _InvertedResidual(Module):
    """MNASNet inverted residual block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        expansion_factor: int,
        bn_momentum: float = 0.1,
    ):
        super().__init__()
        # TODO: Implement inverted residual
        raise NotImplementedError("TODO: Implement _InvertedResidual")


class MNASNet(Module):
    """
    MNASNet model.

    Args:
        alpha: Width multiplier (0.5, 0.75, 1.0, 1.3)
        num_classes: Number of output classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        alpha: float,
        num_classes: int = 1000,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.alpha = alpha

        # TODO: Implement MNASNet
        # See paper for exact architecture
        raise NotImplementedError("TODO: Implement MNASNet")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def mnasnet0_5(num_classes: int = 1000, **kwargs) -> MNASNet:
    """MNASNet with width multiplier 0.5."""
    return MNASNet(0.5, num_classes=num_classes, **kwargs)

def mnasnet0_75(num_classes: int = 1000, **kwargs) -> MNASNet:
    """MNASNet with width multiplier 0.75."""
    return MNASNet(0.75, num_classes=num_classes, **kwargs)

def mnasnet1_0(num_classes: int = 1000, **kwargs) -> MNASNet:
    """MNASNet with width multiplier 1.0."""
    return MNASNet(1.0, num_classes=num_classes, **kwargs)

def mnasnet1_3(num_classes: int = 1000, **kwargs) -> MNASNet:
    """MNASNet with width multiplier 1.3."""
    return MNASNet(1.3, num_classes=num_classes, **kwargs)
