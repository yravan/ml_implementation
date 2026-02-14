"""
Faster R-CNN
============

From "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
https://arxiv.org/abs/1506.01497

Key innovation: Region Proposal Network (RPN) - learns to propose regions.
End-to-end trainable, much faster than R-CNN and Fast R-CNN.

Architecture:
    1. Backbone (e.g., ResNet-FPN) extracts features
    2. RPN proposes regions of interest (RoIs)
    3. RoI pooling extracts fixed-size features from each RoI
    4. Detection head predicts class and refines box

Two-stage detection:
    Stage 1: RPN proposes ~2000 regions
    Stage 2: Fast R-CNN classifies and refines each proposal
"""

from typing import Optional, List, Dict, Tuple
from python.nn_core import Module


class AnchorGenerator(Module):
    """
    Generate anchors for RPN.

    Args:
        sizes: Anchor sizes for each feature map level
        aspect_ratios: Aspect ratios for anchors
    """

    def __init__(
        self,
        sizes: Tuple[Tuple[int, ...], ...] = ((32, 64, 128, 256, 512),),
        aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),),
    ):
        super().__init__()
        # TODO: Implement anchor generation
        raise NotImplementedError("TODO: Implement AnchorGenerator")

    def forward(self, image_list, feature_maps):
        """Generate anchors for the given feature maps."""
        raise NotImplementedError("TODO: Implement forward")


class RPNHead(Module):
    """
    RPN classification and regression heads.

    Args:
        in_channels: Input channels
        num_anchors: Number of anchors per location
    """

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        # TODO: Implement RPN head
        # conv = Conv2d(in_channels, in_channels, 3, padding=1)
        # cls_logits = Conv2d(in_channels, num_anchors, 1)  # objectness
        # bbox_pred = Conv2d(in_channels, num_anchors * 4, 1)  # box deltas
        raise NotImplementedError("TODO: Implement RPNHead")


class RegionProposalNetwork(Module):
    """
    Region Proposal Network.

    Args:
        anchor_generator: Module to generate anchors
        head: RPN head for classification/regression
        fg_iou_thresh: IoU threshold for foreground
        bg_iou_thresh: IoU threshold for background
        batch_size_per_image: Number of anchors to sample per image
        positive_fraction: Fraction of positives in batch
        pre_nms_top_n: Number of proposals before NMS
        post_nms_top_n: Number of proposals after NMS
        nms_thresh: NMS threshold
    """

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: RPNHead,
        fg_iou_thresh: float = 0.7,
        bg_iou_thresh: float = 0.3,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        pre_nms_top_n: Dict[str, int] = None,
        post_nms_top_n: Dict[str, int] = None,
        nms_thresh: float = 0.7,
    ):
        super().__init__()
        # TODO: Implement RPN
        raise NotImplementedError("TODO: Implement RegionProposalNetwork")

    def forward(self, images, features, targets=None):
        """Generate proposals and compute RPN loss if training."""
        raise NotImplementedError("TODO: Implement forward")


class RoIHeads(Module):
    """
    Detection heads that operate on RoI features.

    Args:
        box_roi_pool: RoI pooling module
        box_head: Feature extraction head
        box_predictor: Classification and regression heads
    """

    def __init__(
        self,
        box_roi_pool: Module,
        box_head: Module,
        box_predictor: Module,
        fg_iou_thresh: float = 0.5,
        bg_iou_thresh: float = 0.5,
        batch_size_per_image: int = 512,
        positive_fraction: float = 0.25,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 100,
    ):
        super().__init__()
        # TODO: Implement RoI heads
        raise NotImplementedError("TODO: Implement RoIHeads")


class FasterRCNN(Module):
    """
    Faster R-CNN model.

    Args:
        backbone: Feature extraction backbone
        rpn: Region Proposal Network
        roi_heads: Detection heads
        transform: Image transformation module
    """

    def __init__(
        self,
        backbone: Module,
        rpn: RegionProposalNetwork,
        roi_heads: RoIHeads,
        num_classes: int = 91,
    ):
        super().__init__()
        # TODO: Implement Faster R-CNN
        raise NotImplementedError("TODO: Implement FasterRCNN")

    def forward(self, images, targets=None):
        """
        Forward pass.

        Args:
            images: List of images
            targets: List of target dicts (boxes, labels) for training

        Returns:
            During training: Dict of losses
            During inference: List of detection dicts
        """
        raise NotImplementedError("TODO: Implement forward")


def fasterrcnn_resnet50_fpn(num_classes: int = 91, **kwargs) -> FasterRCNN:
    """Faster R-CNN with ResNet-50-FPN backbone."""
    raise NotImplementedError("TODO: Implement fasterrcnn_resnet50_fpn")


def fasterrcnn_resnet50_fpn_v2(num_classes: int = 91, **kwargs) -> FasterRCNN:
    """Faster R-CNN with ResNet-50-FPN backbone (improved)."""
    raise NotImplementedError("TODO: Implement fasterrcnn_resnet50_fpn_v2")


def fasterrcnn_mobilenet_v3_large_fpn(num_classes: int = 91, **kwargs) -> FasterRCNN:
    """Faster R-CNN with MobileNetV3-Large-FPN backbone."""
    raise NotImplementedError("TODO: Implement fasterrcnn_mobilenet_v3_large_fpn")


def fasterrcnn_mobilenet_v3_large_320_fpn(num_classes: int = 91, **kwargs) -> FasterRCNN:
    """Faster R-CNN with MobileNetV3-Large-FPN backbone (320px input)."""
    raise NotImplementedError("TODO: Implement fasterrcnn_mobilenet_v3_large_320_fpn")
