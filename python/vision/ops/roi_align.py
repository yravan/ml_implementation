"""
RoI Align
=========

Region of Interest Align from Mask R-CNN.
https://arxiv.org/abs/1703.06870

RoI Align improves on RoI Pool by using bilinear interpolation
instead of quantization, avoiding misalignment issues.
"""

import numpy as np
from typing import Tuple, Union


def roi_align(
    input: np.ndarray,
    boxes: np.ndarray,
    output_size: Union[int, Tuple[int, int]],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> np.ndarray:
    """
    Perform RoI Align pooling.

    For each RoI:
    1. Map RoI coordinates to feature map scale
    2. Divide RoI into output_size x output_size bins
    3. For each bin, sample at regular points using bilinear interpolation
    4. Average (or max) the sampled values

    Args:
        input: (N, C, H, W) input feature maps
        boxes: (K, 5) where each row is [batch_idx, x1, y1, x2, y2]
               or list of (Ki, 4) boxes per image
        output_size: (height, width) of output
        spatial_scale: Scale factor from input coords to feature map coords
                      (e.g., 1/16 if feature map is 16x downsampled)
        sampling_ratio: Number of sampling points per bin (default: adaptive)
        aligned: If True, shift coordinates by -0.5 for proper alignment

    Returns:
        (K, C, output_size[0], output_size[1]) pooled features
    """
    raise NotImplementedError("TODO: Implement roi_align")


class RoIAlign:
    """
    RoI Align pooling layer.

    Args:
        output_size: (height, width) of output
        spatial_scale: Scale factor from input to feature map
        sampling_ratio: Number of sampling points per bin
        aligned: Whether to use aligned mode

    Example:
        >>> roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1/16)
        >>> features = roi_align(feature_map, boxes)
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]],
        spatial_scale: float,
        sampling_ratio: int = -1,
        aligned: bool = False,
    ):
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def __call__(
        self,
        input: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RoIAlign")
