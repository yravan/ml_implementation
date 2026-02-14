"""
DeepLabV3
=========

From "Rethinking Atrous Convolution for Semantic Image Segmentation"
https://arxiv.org/abs/1706.05587

Key innovations:
    1. Atrous (dilated) convolution: Expand receptive field without pooling
    2. Atrous Spatial Pyramid Pooling (ASPP): Multi-scale context
    3. Batch normalization in ASPP

ASPP module:
    - 1x1 conv
    - 3x3 conv with dilation 6
    - 3x3 conv with dilation 12
    - 3x3 conv with dilation 18
    - Global average pooling + 1x1 conv
    - Concatenate all + 1x1 conv to fuse

DeepLabV3+ adds a decoder for sharper boundaries.
"""

from typing import List, Optional
from python.nn_core import Module


class ASPPConv(Module):
    """ASPP convolution with given dilation."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        # TODO: Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
        # BatchNorm2d(out_channels)
        # ReLU()
        raise NotImplementedError("TODO: Implement ASPPConv")


class ASPPPooling(Module):
    """Global pooling branch of ASPP."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # TODO: AdaptiveAvgPool2d(1)
        # Conv2d(in_channels, out_channels, 1, bias=False)
        # BatchNorm2d(out_channels)
        # ReLU()
        # Upsample back to input size
        raise NotImplementedError("TODO: Implement ASPPPooling")


class ASPP(Module):
    """
    Atrous Spatial Pyramid Pooling.

    Args:
        in_channels: Input channels
        atrous_rates: Dilation rates for ASPP convolutions
        out_channels: Output channels (default 256)
    """

    def __init__(
        self,
        in_channels: int,
        atrous_rates: List[int] = [6, 12, 18],
        out_channels: int = 256,
    ):
        super().__init__()
        # TODO: Implement ASPP
        # modules = [
        #   Conv1x1(in_channels, out_channels),  # 1x1 conv
        #   ASPPConv(in_channels, out_channels, rates[0]),
        #   ASPPConv(in_channels, out_channels, rates[1]),
        #   ASPPConv(in_channels, out_channels, rates[2]),
        #   ASPPPooling(in_channels, out_channels),
        # ]
        # project = Conv1x1(5 * out_channels, out_channels)
        raise NotImplementedError("TODO: Implement ASPP")

    def forward(self, x):
        """Forward through all branches, concatenate, and project."""
        raise NotImplementedError("TODO: Implement forward")


class DeepLabHead(Module):
    """DeepLab head with ASPP."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # TODO: ASPP(in_channels, [12, 24, 36])
        # Conv2d(256, 256, 3, padding=1, bias=False)
        # BatchNorm2d(256)
        # ReLU()
        # Conv2d(256, num_classes, 1)
        raise NotImplementedError("TODO: Implement DeepLabHead")


class DeepLabV3(Module):
    """
    DeepLabV3 for semantic segmentation.

    Args:
        backbone: Feature extraction backbone
        classifier: DeepLab head
        aux_classifier: Optional auxiliary classifier
    """

    def __init__(
        self,
        backbone: Module,
        classifier: Module,
        aux_classifier: Optional[Module] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        raise NotImplementedError("TODO: Implement DeepLabV3")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def deeplabv3_resnet50(num_classes: int = 21, **kwargs) -> DeepLabV3:
    """DeepLabV3 with ResNet-50 backbone."""
    raise NotImplementedError("TODO: Implement deeplabv3_resnet50")


def deeplabv3_resnet101(num_classes: int = 21, **kwargs) -> DeepLabV3:
    """DeepLabV3 with ResNet-101 backbone."""
    raise NotImplementedError("TODO: Implement deeplabv3_resnet101")


def deeplabv3_mobilenet_v3_large(num_classes: int = 21, **kwargs) -> DeepLabV3:
    """DeepLabV3 with MobileNetV3-Large backbone."""
    raise NotImplementedError("TODO: Implement deeplabv3_mobilenet_v3_large")
