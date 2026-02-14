"""
FCOS - Fully Convolutional One-Stage Object Detection
=====================================================

From "FCOS: Fully Convolutional One-Stage Object Detection"
https://arxiv.org/abs/1904.01355

Key innovation: Anchor-free detection
    - Per-pixel prediction (like semantic segmentation)
    - Predicts: class, centerness, box (l, t, r, b distances)
    - No anchor boxes needed!

Centerness helps suppress low-quality predictions far from object center.

Architecture:
    - FPN backbone
    - Shared head across FPN levels
    - Per-level: classification + centerness + box regression
"""

from typing import List, Optional
from python.nn_core import Module


class FCOSClassificationHead(Module):
    """FCOS classification head."""

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        num_convs: int = 4,
        prior_probability: float = 0.01,
    ):
        super().__init__()
        # TODO: Implement classification head
        raise NotImplementedError("TODO: Implement FCOSClassificationHead")


class FCOSRegressionHead(Module):
    """FCOS regression + centerness head."""

    def __init__(self, in_channels: int, num_anchors: int, num_convs: int = 4):
        super().__init__()
        # TODO: Implement regression head
        # Predicts: box (4 values) + centerness (1 value)
        raise NotImplementedError("TODO: Implement FCOSRegressionHead")


class FCOS(Module):
    """
    FCOS detector.

    Args:
        backbone: Feature extraction backbone with FPN
        num_classes: Number of object classes
        center_sampling_radius: Radius for center sampling
    """

    def __init__(
        self,
        backbone: Module,
        num_classes: int = 91,
        center_sampling_radius: float = 1.5,
        score_thresh: float = 0.2,
        nms_thresh: float = 0.6,
        detections_per_img: int = 100,
        topk_candidates: int = 1000,
    ):
        super().__init__()
        # TODO: Implement FCOS
        raise NotImplementedError("TODO: Implement FCOS")

    def forward(self, images, targets=None):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def fcos_resnet50_fpn(num_classes: int = 91, **kwargs) -> FCOS:
    """FCOS with ResNet-50-FPN backbone."""
    raise NotImplementedError("TODO: Implement fcos_resnet50_fpn")
