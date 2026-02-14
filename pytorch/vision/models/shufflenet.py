"""
ShuffleNet V2
=============

ShuffleNet V2 architecture from "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
https://arxiv.org/abs/1807.11164

Key insights from empirical analysis:
    G1: Equal channel width minimizes memory access cost (MAC)
    G2: Excessive group convolution increases MAC
    G3: Network fragmentation reduces parallelism
    G4: Element-wise operations are non-negligible

Key innovation: Channel split + shuffle
    - Split channels into two branches
    - One branch: identity
    - Other branch: 1x1 conv -> 3x3 depthwise -> 1x1 conv
    - Concatenate and shuffle channels

Variants: x0.5, x1.0, x1.5, x2.0 (width multipliers)
"""

from typing import List
from python.nn_core import Module


def channel_shuffle(x, groups: int):
    """
    Channel shuffle operation.

    Rearranges channels to enable cross-group information flow.
    [g0_c0, g0_c1, g1_c0, g1_c1, ...] -> [g0_c0, g1_c0, g0_c1, g1_c1, ...]
    """
    raise NotImplementedError("TODO: Implement channel_shuffle")


class InvertedResidual(Module):
    """
    ShuffleNet V2 inverted residual block.

    Args:
        inp: Input channels
        oup: Output channels
        stride: Stride (1 or 2)
    """

    def __init__(self, inp: int, oup: int, stride: int):
        super().__init__()
        self.stride = stride

        # TODO: Implement ShuffleNet V2 block
        # If stride == 1:
        #   Split channels in half
        #   Branch 1: identity
        #   Branch 2: 1x1 conv -> 3x3 DW conv -> 1x1 conv
        #   Concat + channel shuffle
        # If stride == 2:
        #   Branch 1: 3x3 DW conv -> 1x1 conv
        #   Branch 2: 1x1 conv -> 3x3 DW conv -> 1x1 conv
        #   Concat + channel shuffle
        raise NotImplementedError("TODO: Implement InvertedResidual")

    def forward(self, x):
        """Forward pass with channel shuffle."""
        raise NotImplementedError("TODO: Implement forward")


class ShuffleNetV2(Module):
    """
    ShuffleNet V2 model.

    Args:
        stages_repeats: Number of blocks in each stage
        stages_out_channels: Output channels for each stage
        num_classes: Number of output classes
    """

    def __init__(
        self,
        stages_repeats: List[int],
        stages_out_channels: List[int],
        num_classes: int = 1000,
    ):
        super().__init__()
        # TODO: Implement ShuffleNetV2
        raise NotImplementedError("TODO: Implement ShuffleNetV2")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def shufflenet_v2_x0_5(num_classes: int = 1000, **kwargs) -> ShuffleNetV2:
    """ShuffleNet V2 x0.5 model."""
    return ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], num_classes=num_classes, **kwargs)

def shufflenet_v2_x1_0(num_classes: int = 1000, **kwargs) -> ShuffleNetV2:
    """ShuffleNet V2 x1.0 model."""
    return ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], num_classes=num_classes, **kwargs)

def shufflenet_v2_x1_5(num_classes: int = 1000, **kwargs) -> ShuffleNetV2:
    """ShuffleNet V2 x1.5 model."""
    return ShuffleNetV2([4, 8, 4], [24, 176, 352, 704, 1024], num_classes=num_classes, **kwargs)

def shufflenet_v2_x2_0(num_classes: int = 1000, **kwargs) -> ShuffleNetV2:
    """ShuffleNet V2 x2.0 model."""
    return ShuffleNetV2([4, 8, 4], [24, 244, 488, 976, 2048], num_classes=num_classes, **kwargs)
