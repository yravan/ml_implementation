"""
SqueezeNet
==========

SqueezeNet model architectures from "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters"
https://arxiv.org/abs/1602.07360

Key innovation: Fire modules that "squeeze" then "expand"
    - Squeeze: 1x1 conv to reduce channels
    - Expand: parallel 1x1 and 3x3 convs, concatenated

Design strategies:
    1. Replace 3x3 filters with 1x1 filters (9x fewer parameters)
    2. Decrease input channels to 3x3 filters (squeeze layers)
    3. Downsample late in the network (larger activation maps)

Variants:
    SqueezeNet 1.0: Original architecture
    SqueezeNet 1.1: 2.4x less computation, same accuracy
"""

from python.nn_core import Module


class Fire(Module):
    """
    Fire module: squeeze then expand.

    Args:
        inplanes: Input channels
        squeeze_planes: Squeeze layer output channels
        expand1x1_planes: Expand 1x1 output channels
        expand3x3_planes: Expand 3x3 output channels
    """

    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
    ):
        super().__init__()
        # TODO: Implement Fire module
        # squeeze = Conv2d(inplanes, squeeze_planes, 1)
        # squeeze_activation = ReLU()
        # expand1x1 = Conv2d(squeeze_planes, expand1x1_planes, 1)
        # expand1x1_activation = ReLU()
        # expand3x3 = Conv2d(squeeze_planes, expand3x3_planes, 3, padding=1)
        # expand3x3_activation = ReLU()
        raise NotImplementedError("TODO: Implement Fire module")

    def forward(self, x):
        """
        Forward pass.
        Squeeze -> (Expand1x1 || Expand3x3) -> Concat
        """
        raise NotImplementedError("TODO: Implement forward")


class SqueezeNet(Module):
    """
    SqueezeNet model.

    Args:
        version: '1_0' or '1_1'
        num_classes: Number of output classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        version: str = '1_0',
        num_classes: int = 1000,
        dropout: float = 0.5,
    ):
        super().__init__()
        # TODO: Implement SqueezeNet
        # Version 1.0:
        #   Conv2d(3, 96, 7, stride=2)
        #   MaxPool2d(3, stride=2)
        #   Fire(96, 16, 64, 64)
        #   Fire(128, 16, 64, 64)
        #   Fire(128, 32, 128, 128)
        #   MaxPool2d(3, stride=2)
        #   Fire(256, 32, 128, 128)
        #   Fire(256, 48, 192, 192)
        #   Fire(384, 48, 192, 192)
        #   Fire(384, 64, 256, 256)
        #   MaxPool2d(3, stride=2)
        #   Fire(512, 64, 256, 256)
        #   Dropout(dropout)
        #   Conv2d(512, num_classes, 1)
        #   AdaptiveAvgPool2d(1)
        raise NotImplementedError("TODO: Implement SqueezeNet")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def squeezenet1_0(num_classes: int = 1000, **kwargs) -> SqueezeNet:
    """SqueezeNet 1.0 model."""
    return SqueezeNet('1_0', num_classes=num_classes, **kwargs)

def squeezenet1_1(num_classes: int = 1000, **kwargs) -> SqueezeNet:
    """SqueezeNet 1.1 model (2.4x less computation)."""
    return SqueezeNet('1_1', num_classes=num_classes, **kwargs)
