"""
Mask R-CNN
==========

From "Mask R-CNN" https://arxiv.org/abs/1703.06870

Extends Faster R-CNN with a mask prediction branch for instance segmentation.

Key additions:
    1. Mask head: Predicts binary mask for each RoI
    2. RoIAlign: Improved RoI pooling with bilinear interpolation

Architecture:
    Faster R-CNN backbone + RPN + detection heads
    + Mask head (FCN applied to each RoI)
"""

from typing import Optional
from python.nn_core import Module
from .faster_rcnn import FasterRCNN


class MaskRCNNHeads(Module):
    """
    Mask prediction head.

    Args:
        in_channels: Input channels
        layers: Channel sizes for conv layers
        dilation: Dilation for convolutions
    """

    def __init__(
        self,
        in_channels: int,
        layers: tuple = (256, 256, 256, 256),
        dilation: int = 1,
    ):
        super().__init__()
        # TODO: Implement mask head
        # Series of Conv2d-ReLU layers
        raise NotImplementedError("TODO: Implement MaskRCNNHeads")


class MaskRCNNPredictor(Module):
    """
    Mask predictor head.

    Args:
        in_channels: Input channels
        dim_reduced: Reduced dimension
        num_classes: Number of classes
    """

    def __init__(self, in_channels: int, dim_reduced: int, num_classes: int):
        super().__init__()
        # TODO: Implement mask predictor
        # ConvTranspose2d for upsampling
        # Conv2d for final prediction
        raise NotImplementedError("TODO: Implement MaskRCNNPredictor")


class MaskRCNN(FasterRCNN):
    """
    Mask R-CNN for instance segmentation.

    Extends Faster R-CNN with mask prediction.
    """

    def __init__(
        self,
        backbone: Module,
        num_classes: int = 91,
        # ... Faster R-CNN params
        mask_roi_pool: Optional[Module] = None,
        mask_head: Optional[Module] = None,
        mask_predictor: Optional[Module] = None,
    ):
        # TODO: Implement Mask R-CNN
        raise NotImplementedError("TODO: Implement MaskRCNN")


def maskrcnn_resnet50_fpn(num_classes: int = 91, **kwargs) -> MaskRCNN:
    """Mask R-CNN with ResNet-50-FPN backbone."""
    raise NotImplementedError("TODO: Implement maskrcnn_resnet50_fpn")


def maskrcnn_resnet50_fpn_v2(num_classes: int = 91, **kwargs) -> MaskRCNN:
    """Mask R-CNN with ResNet-50-FPN backbone (improved)."""
    raise NotImplementedError("TODO: Implement maskrcnn_resnet50_fpn_v2")
