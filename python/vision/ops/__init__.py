"""
Vision Operations
=================

Low-level operations for computer vision.
"""

from .boxes import (
    box_area,
    box_convert,
    box_iou,
    generalized_box_iou,
    complete_box_iou,
    distance_box_iou,
    nms,
    batched_nms,
    remove_small_boxes,
    clip_boxes_to_image,
)

from .roi_align import (
    roi_align,
    RoIAlign,
)

from .roi_pool import (
    roi_pool,
    RoIPool,
)

from .poolers import (
    MultiScaleRoIAlign,
)

from .focal_loss import (
    sigmoid_focal_loss,
)

from .deform_conv import (
    deform_conv2d,
    DeformConv2d,
)

from .misc import (
    FrozenBatchNorm2d,
    Conv2dNormActivation,
    SqueezeExcitation,
    StochasticDepth,
    MLP,
)

from .feature_pyramid_network import (
    FeaturePyramidNetwork,
)


__all__ = [
    # Boxes
    'box_area',
    'box_convert',
    'box_iou',
    'generalized_box_iou',
    'complete_box_iou',
    'distance_box_iou',
    'nms',
    'batched_nms',
    'remove_small_boxes',
    'clip_boxes_to_image',
    # RoI operations
    'roi_align',
    'RoIAlign',
    'roi_pool',
    'RoIPool',
    'MultiScaleRoIAlign',
    # Loss
    'sigmoid_focal_loss',
    # Deformable conv
    'deform_conv2d',
    'DeformConv2d',
    # Building blocks
    'FrozenBatchNorm2d',
    'Conv2dNormActivation',
    'SqueezeExcitation',
    'StochasticDepth',
    'MLP',
    'FeaturePyramidNetwork',
]
