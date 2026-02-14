"""
Feature Pyramid Network
=======================

FPN from "Feature Pyramid Networks for Object Detection".
https://arxiv.org/abs/1612.03144

FPN builds a multi-scale feature pyramid with strong semantics at all scales.
"""

import numpy as np
from typing import Dict, List, Optional, Callable


class FeaturePyramidNetwork:
    """
    Feature Pyramid Network.

    Takes multi-scale features from a backbone (e.g., ResNet stages)
    and builds a feature pyramid with:
    1. 1x1 conv to reduce channels (lateral connections)
    2. Top-down pathway with upsampling and addition
    3. 3x3 conv to reduce aliasing effects

    Architecture:
    ```
    C5 (small) ----1x1----> P5 ---------> P5_out
                            |
                            v (upsample + add)
    C4 ---------1x1----> P4 ---------> P4_out
                            |
                            v (upsample + add)
    C3 ---------1x1----> P3 ---------> P3_out
                            |
                            v (upsample + add)
    C2 (large) --1x1----> P2 ---------> P2_out
    ```

    Args:
        in_channels_list: Number of channels for each input feature
                         (ordered from largest to smallest resolution)
        out_channels: Number of output channels (same for all levels)
        extra_blocks: Optional module to add extra levels (e.g., P6, P7)
        norm_layer: Optional normalization layer

    Example:
        >>> # For ResNet backbone with stages C2, C3, C4, C5
        >>> fpn = FeaturePyramidNetwork(
        ...     in_channels_list=[256, 512, 1024, 2048],
        ...     out_channels=256
        ... )
        >>> features = {'feat0': c2, 'feat1': c3, 'feat2': c4, 'feat3': c5}
        >>> pyramid = fpn(features)
        >>> # pyramid has same keys with all channels = 256
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[Callable] = None,
        norm_layer: Optional[Callable] = None,
    ):
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.extra_blocks = extra_blocks

        # Lateral connections (1x1 convs to match channel dimensions)
        self.inner_blocks = []  # One per input feature

        # Output convolutions (3x3 convs to reduce aliasing)
        self.layer_blocks = []  # One per output feature

    def __call__(
        self,
        x: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Build feature pyramid.

        Args:
            x: Dict mapping names to feature maps
               Should be ordered from largest to smallest resolution

        Returns:
            Dict mapping names to pyramid features
            All outputs have out_channels channels
        """
        raise NotImplementedError("TODO: Implement FeaturePyramidNetwork")


class LastLevelMaxPool:
    """
    Adds an extra level using max pooling on the last feature.

    Used in Faster R-CNN to add P6 from P5.
    """

    def __call__(self, x: np.ndarray, names: List[str]) -> np.ndarray:
        raise NotImplementedError("TODO: Implement LastLevelMaxPool")


class LastLevelP6P7:
    """
    Adds P6 and P7 levels for RetinaNet.

    P6 = Conv3x3(P5) or Conv3x3(C5) with stride 2
    P7 = Conv3x3(ReLU(P6)) with stride 2

    Args:
        in_channels: Input channels (from P5 or C5)
        out_channels: Output channels
        use_p5: If True, use P5 as input; else use C5
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_p5: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_p5 = use_p5

        self.p6 = None  # Conv3x3 stride 2
        self.p7 = None  # Conv3x3 stride 2

    def __call__(
        self,
        p5: np.ndarray,
        c5: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Generate P6 and P7.

        Returns:
            [P6, P7] feature maps
        """
        raise NotImplementedError("TODO: Implement LastLevelP6P7")
