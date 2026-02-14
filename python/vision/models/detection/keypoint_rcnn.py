"""
Keypoint R-CNN
==============

Extends Mask R-CNN for human pose estimation (keypoint detection).

Predicts K keypoint locations for each detected person.
COCO keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
"""

from python.nn_core import Module
from .mask_rcnn import MaskRCNN


class KeypointRCNN(MaskRCNN):
    """
    Keypoint R-CNN for pose estimation.

    Extends Mask R-CNN with keypoint prediction head.
    """

    def __init__(
        self,
        backbone: Module,
        num_classes: int = 2,  # person + background
        num_keypoints: int = 17,  # COCO keypoints
        **kwargs
    ):
        # TODO: Implement Keypoint R-CNN
        raise NotImplementedError("TODO: Implement KeypointRCNN")


def keypointrcnn_resnet50_fpn(num_classes: int = 2, num_keypoints: int = 17, **kwargs) -> KeypointRCNN:
    """Keypoint R-CNN with ResNet-50-FPN backbone."""
    raise NotImplementedError("TODO: Implement keypointrcnn_resnet50_fpn")
