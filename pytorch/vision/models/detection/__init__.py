"""
Object Detection Models
=======================

Models for object detection and instance segmentation.

Two-stage detectors:
- Faster R-CNN: Region Proposal Network + RoI heads
- Mask R-CNN: Faster R-CNN + mask prediction
- Keypoint R-CNN: Faster R-CNN + keypoint prediction

One-stage detectors:
- RetinaNet: Feature Pyramid Network + Focal Loss
- SSD: Single Shot MultiBox Detector
- FCOS: Fully Convolutional One-Stage detector
"""

from .faster_rcnn import (
    fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN
)
from .mask_rcnn import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2, MaskRCNN
from .keypoint_rcnn import keypointrcnn_resnet50_fpn, KeypointRCNN
from .retinanet import retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2, RetinaNet
from .ssd import ssd300_vgg16, ssdlite320_mobilenet_v3_large, SSD
from .fcos import fcos_resnet50_fpn, FCOS

__all__ = [
    'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2',
    'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_mobilenet_v3_large_320_fpn',
    'FasterRCNN',
    'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 'MaskRCNN',
    'keypointrcnn_resnet50_fpn', 'KeypointRCNN',
    'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 'RetinaNet',
    'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 'SSD',
    'fcos_resnet50_fpn', 'FCOS',
]
