"""
Video ResNet
============

3D ResNet variants for video classification.

From various papers:
- R3D: "Learning Spatio-Temporal Features with 3D Residual Networks"
- MC3: "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
- R(2+1)D: Same as MC3, factorized 3D convs

R3D-18: Standard 3D ResNet-18
MC3-18: Mixed 2D spatial + 3D temporal convolutions
R(2+1)D-18: Factorized 3D convolutions (2D spatial + 1D temporal)
"""

from typing import Tuple, Optional, Callable
from python.nn_core import Module


class Conv3DSimple(Module):
    """Standard 3D convolution."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        midplanes: int = None,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        # TODO: Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), ...)
        raise NotImplementedError("TODO: Implement Conv3DSimple")


class Conv2Plus1D(Module):
    """
    (2+1)D convolution: Factorized 3D conv into 2D spatial + 1D temporal.

    More parameters but easier to optimize than full 3D conv.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        midplanes: int,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        # TODO: Implement (2+1)D convolution
        # spatial_conv = Conv3d(in_planes, midplanes, (1, 3, 3), ...)
        # temporal_conv = Conv3d(midplanes, out_planes, (3, 1, 1), ...)
        raise NotImplementedError("TODO: Implement Conv2Plus1D")


class Conv3DNoTemporal(Module):
    """MC3 block: 3D spatial conv with no temporal extent (2D effectively)."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        midplanes: int = None,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        # TODO: Conv3d with kernel (1, 3, 3)
        raise NotImplementedError("TODO: Implement Conv3DNoTemporal")


class BasicBlock3D(Module):
    """Basic 3D residual block."""
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable,
        stride: int = 1,
        downsample: Optional[Module] = None,
    ):
        super().__init__()
        # TODO: Implement 3D basic block
        raise NotImplementedError("TODO: Implement BasicBlock3D")


class Bottleneck3D(Module):
    """Bottleneck 3D residual block."""
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable,
        stride: int = 1,
        downsample: Optional[Module] = None,
    ):
        super().__init__()
        # TODO: Implement 3D bottleneck
        raise NotImplementedError("TODO: Implement Bottleneck3D")


class VideoResNet(Module):
    """
    Video ResNet (3D ResNet variants).

    Args:
        block: Block type (BasicBlock3D or Bottleneck3D)
        conv_makers: Conv builders for each layer
        layers: Number of blocks in each layer
        stem: Stem module
        num_classes: Number of output classes
    """

    def __init__(
        self,
        block,
        conv_makers,
        layers,
        stem,
        num_classes: int = 400,
    ):
        super().__init__()
        # TODO: Implement VideoResNet
        raise NotImplementedError("TODO: Implement VideoResNet")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Video tensor of shape (B, C, T, H, W)
        """
        raise NotImplementedError("TODO: Implement forward")


def r3d_18(num_classes: int = 400, **kwargs) -> VideoResNet:
    """R3D-18: 3D ResNet-18 with full 3D convolutions."""
    raise NotImplementedError("TODO: Implement r3d_18")


def mc3_18(num_classes: int = 400, **kwargs) -> VideoResNet:
    """MC3-18: Mixed 3D convolutions (2D early, 3D late)."""
    raise NotImplementedError("TODO: Implement mc3_18")


def r2plus1d_18(num_classes: int = 400, **kwargs) -> VideoResNet:
    """R(2+1)D-18: Factorized 3D convolutions."""
    raise NotImplementedError("TODO: Implement r2plus1d_18")
