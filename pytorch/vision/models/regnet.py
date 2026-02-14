"""
RegNet
======

RegNet architecture from "Designing Network Design Spaces"
https://arxiv.org/abs/2003.13678

Key innovation: Systematic network design through design space exploration.
Instead of designing individual networks, design the space of possible networks.

The RegNet design space is parameterized by:
    - d: depth (number of blocks)
    - w_0: initial width
    - w_a: slope for width progression
    - w_m: multiplier for width quantization
    - g: group width for grouped convolution

Width at block i: w_i = w_0 + w_a * i, then quantize to nearest multiple of 8.

Two variants:
    - RegNetX: Standard blocks
    - RegNetY: Blocks with Squeeze-and-Excitation
"""

from typing import List, Optional, Tuple
from python.nn_core import Module


class SimpleStemIN(Module):
    """Simple stem for RegNet (Conv-BN-ReLU)."""

    def __init__(self, width_in: int, width_out: int):
        super().__init__()
        # TODO: Conv2d(width_in, width_out, 3, stride=2, padding=1)
        # BatchNorm2d, ReLU
        raise NotImplementedError("TODO: Implement SimpleStemIN")


class BottleneckTransform(Module):
    """Bottleneck transformation for RegNet: 1x1 -> 3x3 -> 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ):
        super().__init__()
        # TODO: Implement bottleneck transform
        raise NotImplementedError("TODO: Implement BottleneckTransform")


class ResBottleneckBlock(Module):
    """Residual bottleneck block for RegNet."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ):
        super().__init__()
        # TODO: Implement residual block with optional SE
        raise NotImplementedError("TODO: Implement ResBottleneckBlock")


class AnyStage(Module):
    """Stage containing multiple blocks."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ):
        super().__init__()
        # TODO: Implement stage
        raise NotImplementedError("TODO: Implement AnyStage")


class RegNet(Module):
    """
    RegNet model.

    Args:
        block_params: Configuration for each stage
        num_classes: Number of output classes
        stem_width: Width of stem
    """

    def __init__(
        self,
        block_params: List[Tuple],
        num_classes: int = 1000,
        stem_width: int = 32,
    ):
        super().__init__()
        # TODO: Implement RegNet
        raise NotImplementedError("TODO: Implement RegNet")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


# RegNetX variants (without SE)
def regnet_x_400mf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetX-400MF (~400 MFLOPs)."""
    raise NotImplementedError("TODO: Implement regnet_x_400mf")

def regnet_x_800mf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetX-800MF."""
    raise NotImplementedError("TODO: Implement regnet_x_800mf")

def regnet_x_1_6gf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetX-1.6GF."""
    raise NotImplementedError("TODO: Implement regnet_x_1_6gf")

def regnet_x_3_2gf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetX-3.2GF."""
    raise NotImplementedError("TODO: Implement regnet_x_3_2gf")

def regnet_x_8gf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetX-8GF."""
    raise NotImplementedError("TODO: Implement regnet_x_8gf")

def regnet_x_16gf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetX-16GF."""
    raise NotImplementedError("TODO: Implement regnet_x_16gf")

def regnet_x_32gf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetX-32GF."""
    raise NotImplementedError("TODO: Implement regnet_x_32gf")

# RegNetY variants (with SE)
def regnet_y_400mf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetY-400MF (~400 MFLOPs, with SE)."""
    raise NotImplementedError("TODO: Implement regnet_y_400mf")

def regnet_y_800mf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetY-800MF."""
    raise NotImplementedError("TODO: Implement regnet_y_800mf")

def regnet_y_1_6gf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetY-1.6GF."""
    raise NotImplementedError("TODO: Implement regnet_y_1_6gf")

def regnet_y_3_2gf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetY-3.2GF."""
    raise NotImplementedError("TODO: Implement regnet_y_3_2gf")

def regnet_y_8gf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetY-8GF."""
    raise NotImplementedError("TODO: Implement regnet_y_8gf")

def regnet_y_16gf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetY-16GF."""
    raise NotImplementedError("TODO: Implement regnet_y_16gf")

def regnet_y_32gf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetY-32GF."""
    raise NotImplementedError("TODO: Implement regnet_y_32gf")

def regnet_y_128gf(num_classes: int = 1000, **kwargs) -> RegNet:
    """RegNetY-128GF."""
    raise NotImplementedError("TODO: Implement regnet_y_128gf")
