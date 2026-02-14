"""
LRASPP - Lite Reduced Atrous Spatial Pyramid Pooling
====================================================

From "Searching for MobileNetV3" https://arxiv.org/abs/1905.02244

A lightweight segmentation head designed for mobile/edge deployment.
Much faster than DeepLabV3 while maintaining reasonable accuracy.

Key design:
    - Simplified ASPP: 1x1 conv + global pooling branch only
    - Low-level features from earlier backbone stage
    - Efficient for real-time segmentation
"""

from typing import Optional
from python.nn_core import Module


class LRASPP(Module):
    """
    Lite R-ASPP segmentation model.

    Args:
        backbone: Feature extraction backbone (MobileNetV3)
        low_channels: Channels from low-level features
        high_channels: Channels from high-level features
        num_classes: Number of output classes
        inter_channels: Intermediate channels (default 128)
    """

    def __init__(
        self,
        backbone: Module,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int = 128,
    ):
        super().__init__()
        # TODO: Implement LRASPP
        # High-level branch:
        #   Conv2d(high_channels, inter_channels, 1, bias=False)
        #   BatchNorm2d(inter_channels)
        #   ReLU()
        # Global pooling branch:
        #   AdaptiveAvgPool2d(1)
        #   Conv2d(high_channels, inter_channels, 1, bias=False)
        #   Sigmoid()
        # Low-level branch:
        #   Conv2d(low_channels, inter_channels, 1, bias=False)
        #   BatchNorm2d(inter_channels)
        #   ReLU()
        # Classifier:
        #   Conv2d(inter_channels, num_classes, 1)
        raise NotImplementedError("TODO: Implement LRASPP")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def lraspp_mobilenet_v3_large(num_classes: int = 21, **kwargs) -> LRASPP:
    """LRASPP with MobileNetV3-Large backbone."""
    raise NotImplementedError("TODO: Implement lraspp_mobilenet_v3_large")
