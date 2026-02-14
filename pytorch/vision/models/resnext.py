"""
ResNeXt
=======

ResNeXt model architectures from "Aggregated Residual Transformations for Deep Networks"
https://arxiv.org/abs/1611.05431

Key innovation: Grouped convolutions with cardinality (number of groups).
Instead of going deeper or wider, ResNeXt increases the cardinality dimension.

The block structure: split-transform-merge
    - Split input into groups
    - Apply transformation to each group
    - Merge by concatenation + 1x1 conv

Notation: ResNeXt-depth_cardinalityxwidth
    - ResNeXt-50_32x4d: 50 layers, 32 groups, 4 channels per group
    - ResNeXt-101_32x8d: 101 layers, 32 groups, 8 channels per group
"""

from python.nn_core import Module
from .resnet import ResNet, Bottleneck


class ResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck with grouped convolutions.

    Args:
        inplanes: Input channels
        planes: Output channels (before expansion)
        stride: Stride for 3x3 conv
        downsample: Downsample for skip connection
        groups: Number of groups (cardinality)
        base_width: Base width per group
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 32,
        base_width: int = 4,
    ):
        # TODO: Implement ResNeXt bottleneck
        # Width = planes * (base_width / 64) * groups
        # conv1 = Conv2d(inplanes, width, 1, bias=False)
        # bn1 = BatchNorm2d(width)
        # conv2 = Conv2d(width, width, 3, stride, padding=1, groups=groups, bias=False)
        # bn2 = BatchNorm2d(width)
        # conv3 = Conv2d(width, planes * self.expansion, 1, bias=False)
        # bn3 = BatchNorm2d(planes * self.expansion)
        raise NotImplementedError("TODO: Implement ResNeXtBottleneck")


def resnext50_32x4d(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNeXt-50 32x4d model."""
    # TODO: Return ResNet with groups=32, width_per_group=4
    raise NotImplementedError("TODO: Implement resnext50_32x4d")


def resnext101_32x8d(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNeXt-101 32x8d model."""
    raise NotImplementedError("TODO: Implement resnext101_32x8d")


def resnext101_64x4d(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNeXt-101 64x4d model."""
    raise NotImplementedError("TODO: Implement resnext101_64x4d")
