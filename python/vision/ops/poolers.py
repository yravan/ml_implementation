"""
Feature Poolers
===============

Multi-scale feature pooling for object detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class MultiScaleRoIAlign:
    """
    Multi-scale RoI Align for Feature Pyramid Networks.

    Pools RoIs from appropriate pyramid levels based on RoI size.
    Uses the formula from FPN paper to assign RoIs to levels:

    level = floor(k0 + log2(sqrt(w*h) / 224))

    where k0 is the target level for a 224x224 box (typically 4).

    Args:
        featmap_names: Names of feature maps to pool from (e.g., ['0', '1', '2', '3'])
        output_size: (height, width) of output
        sampling_ratio: Sampling points per bin for RoI Align
        canonical_scale: Reference scale (default: 224)
        canonical_level: Level for canonical_scale boxes (default: 4)

    Example:
        >>> pooler = MultiScaleRoIAlign(
        ...     featmap_names=['feat0', 'feat1', 'feat2', 'feat3'],
        ...     output_size=7,
        ...     sampling_ratio=2
        ... )
        >>> features = pooler(feature_dict, boxes, image_sizes)
    """

    def __init__(
        self,
        featmap_names: List[str],
        output_size: int,
        sampling_ratio: int,
        canonical_scale: int = 224,
        canonical_level: int = 4,
    ):
        self.featmap_names = featmap_names
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

    def __call__(
        self,
        x: Dict[str, np.ndarray],
        boxes: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Pool features from multiple scales.

        Args:
            x: Dict mapping feature names to (N, C, H, W) feature maps
            boxes: List of (Ki, 4) boxes per image
            image_shapes: List of (H, W) original image sizes

        Returns:
            (sum(Ki), C, output_size, output_size) pooled features
        """
        raise NotImplementedError("TODO: Implement MultiScaleRoIAlign")
