"""
MobileNet
=========

MobileNet architectures for efficient inference on mobile/edge devices.

MobileNetV2: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
https://arxiv.org/abs/1801.04381

Key innovation: Inverted residual blocks
    - Expand channels with 1x1 conv
    - Depthwise 3x3 conv
    - Project back to low dimension with 1x1 conv (linear, no ReLU!)
    - Skip connection on the narrow layers

MobileNetV3: "Searching for MobileNetV3"
https://arxiv.org/abs/1905.02244

Improvements:
    - Neural Architecture Search (NAS)
    - Squeeze-and-Excitation (SE) blocks
    - Hard-swish activation: x * ReLU6(x + 3) / 6
    - Redesigned expensive layers
"""

from typing import List, Optional, Callable
from python.nn_core import Module


class ConvBNActivation(Module):
    """Conv2d + BatchNorm + Activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[Module] = None,
    ):
        super().__init__()
        # TODO: Implement ConvBNActivation
        raise NotImplementedError("TODO: Implement ConvBNActivation")


class InvertedResidual(Module):
    """
    MobileNetV2 inverted residual block.

    Args:
        inp: Input channels
        oup: Output channels
        stride: Stride for depthwise conv
        expand_ratio: Expansion ratio for hidden dim
    """

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.use_res_connect = stride == 1 and inp == oup

        # TODO: Implement inverted residual
        # hidden_dim = inp * expand_ratio
        # If expand_ratio != 1:
        #   Conv 1x1 (expand)
        # Depthwise Conv 3x3
        # Conv 1x1 (project) - NO activation!
        raise NotImplementedError("TODO: Implement InvertedResidual")

    def forward(self, x):
        """Forward with residual connection."""
        raise NotImplementedError("TODO: Implement forward")


class SqueezeExcitation(Module):
    """
    Squeeze-and-Excitation block for MobileNetV3.

    Args:
        input_channels: Input channels
        squeeze_channels: Squeezed channels
    """

    def __init__(self, input_channels: int, squeeze_channels: int):
        super().__init__()
        # TODO: Implement SE block
        # AdaptiveAvgPool2d(1)
        # Conv2d(input_channels, squeeze_channels, 1)
        # ReLU()
        # Conv2d(squeeze_channels, input_channels, 1)
        # Sigmoid() or Hardsigmoid()
        raise NotImplementedError("TODO: Implement SqueezeExcitation")


class MobileNetV2(Module):
    """
    MobileNet V2 model.

    Args:
        num_classes: Number of output classes
        width_mult: Width multiplier (0.5, 0.75, 1.0, etc.)
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        # TODO: Implement MobileNetV2
        # See paper for inverted_residual_setting
        raise NotImplementedError("TODO: Implement MobileNetV2")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


class MobileNetV3(Module):
    """
    MobileNet V3 model.

    Args:
        inverted_residual_setting: Network architecture config
        num_classes: Number of output classes
        width_mult: Width multiplier
        dropout: Dropout probability
    """

    def __init__(
        self,
        inverted_residual_setting: List,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        # TODO: Implement MobileNetV3
        raise NotImplementedError("TODO: Implement MobileNetV3")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def mobilenet_v2(num_classes: int = 1000, **kwargs) -> MobileNetV2:
    """MobileNet V2 model."""
    return MobileNetV2(num_classes=num_classes, **kwargs)

def mobilenet_v3_small(num_classes: int = 1000, **kwargs) -> MobileNetV3:
    """MobileNet V3 Small model."""
    # TODO: Define small architecture config
    raise NotImplementedError("TODO: Implement mobilenet_v3_small")

def mobilenet_v3_large(num_classes: int = 1000, **kwargs) -> MobileNetV3:
    """MobileNet V3 Large model."""
    # TODO: Define large architecture config
    raise NotImplementedError("TODO: Implement mobilenet_v3_large")
