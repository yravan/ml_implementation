"""
RoI Pool
========

Region of Interest Pooling from Fast R-CNN.
https://arxiv.org/abs/1504.08083

RoI Pool extracts fixed-size features from arbitrary-sized regions.
Note: RoI Align is preferred in modern detectors due to better alignment.
"""

import numpy as np
from typing import Tuple, Union


def roi_pool(
    input: np.ndarray,
    boxes: np.ndarray,
    output_size: Union[int, Tuple[int, int]],
    spatial_scale: float = 1.0,
) -> np.ndarray:
    """
    Perform RoI Pooling.

    For each RoI:
    1. Map RoI coordinates to feature map scale (with quantization)
    2. Divide RoI into output_size x output_size bins
    3. Max pool within each bin

    Args:
        input: (N, C, H, W) input feature maps
        boxes: (K, 5) where each row is [batch_idx, x1, y1, x2, y2]
        output_size: (height, width) of output
        spatial_scale: Scale factor from input to feature map coords

    Returns:
        (K, C, output_size[0], output_size[1]) pooled features
    """
    raise NotImplementedError("TODO: Implement roi_pool")


class RoIPool:
    """
    RoI Pooling layer.

    Args:
        output_size: (height, width) of output
        spatial_scale: Scale factor from input to feature map

    Example:
        >>> roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1/16)
        >>> features = roi_pool(feature_map, boxes)
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]],
        spatial_scale: float,
    ):
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def __call__(
        self,
        input: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RoIPool")
