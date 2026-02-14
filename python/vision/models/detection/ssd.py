"""
SSD - Single Shot MultiBox Detector
===================================

From "SSD: Single Shot MultiBox Detector"
https://arxiv.org/abs/1512.02325

Key ideas:
    1. Single-shot: No proposal generation, direct detection
    2. Multi-scale detection: Predict at multiple feature map scales
    3. Default boxes: Pre-defined anchor boxes at each location

SSD300: 300x300 input, VGG16 backbone
SSDLite: Lightweight version with MobileNet backbone
"""

from typing import List, Dict, Optional
from python.nn_core import Module


class SSDHead(Module):
    """
    SSD detection head.

    Args:
        in_channels: Input channels for each feature level
        num_anchors: Number of anchors per location at each level
        num_classes: Number of classes
    """

    def __init__(
        self,
        in_channels: List[int],
        num_anchors: List[int],
        num_classes: int,
    ):
        super().__init__()
        # TODO: Implement SSD head
        # For each feature level:
        #   classification_head = Conv2d(in_ch, num_anchors * num_classes, 3, padding=1)
        #   regression_head = Conv2d(in_ch, num_anchors * 4, 3, padding=1)
        raise NotImplementedError("TODO: Implement SSDHead")


class SSD(Module):
    """
    SSD detector.

    Args:
        backbone: Feature extraction backbone
        anchor_generator: Anchor generation module
        head: Detection head
        num_classes: Number of classes
    """

    def __init__(
        self,
        backbone: Module,
        anchor_generator: Module,
        head: Module,
        num_classes: int = 91,
        score_thresh: float = 0.01,
        nms_thresh: float = 0.45,
        detections_per_img: int = 200,
        iou_thresh: float = 0.5,
        neg_pos_ratio: float = 3.0,
    ):
        super().__init__()
        # TODO: Implement SSD
        raise NotImplementedError("TODO: Implement SSD")

    def forward(self, images, targets=None):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def ssd300_vgg16(num_classes: int = 91, **kwargs) -> SSD:
    """SSD300 with VGG16 backbone."""
    raise NotImplementedError("TODO: Implement ssd300_vgg16")


def ssdlite320_mobilenet_v3_large(num_classes: int = 91, **kwargs) -> SSD:
    """SSDLite with MobileNetV3-Large backbone (320px input)."""
    raise NotImplementedError("TODO: Implement ssdlite320_mobilenet_v3_large")
