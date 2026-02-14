"""
DenseNet
========

DenseNet model architectures from "Densely Connected Convolutional Networks"
https://arxiv.org/abs/1608.06993

Key innovation: Dense connections - each layer receives input from ALL preceding layers.
This creates shorter connections between layers, encouraging feature reuse.

For a layer l:
    x_l = H_l([x_0, x_1, ..., x_{l-1}])

where [x_0, x_1, ...] denotes concatenation of feature maps.

Growth rate (k): Each layer produces k feature maps, so layer l has
    k_0 + k * (l - 1) input feature maps

Variants:
    DenseNet-121: [6, 12, 24, 16] blocks, k=32
    DenseNet-161: [6, 12, 36, 24] blocks, k=48
    DenseNet-169: [6, 12, 32, 32] blocks, k=32
    DenseNet-201: [6, 12, 48, 32] blocks, k=32
"""

from typing import List, Tuple
from python.nn_core import Module


class _DenseLayer(Module):
    """
    Single dense layer: BN -> ReLU -> Conv1x1 -> BN -> ReLU -> Conv3x3

    Args:
        num_input_features: Number of input channels
        growth_rate: Number of output channels (k)
        bn_size: Bottleneck size multiplier (output of 1x1 = bn_size * growth_rate)
        drop_rate: Dropout rate
    """

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement dense layer
        # bn1 = BatchNorm2d(num_input_features)
        # relu1 = ReLU()
        # conv1 = Conv2d(num_input_features, bn_size * growth_rate, 1, bias=False)
        # bn2 = BatchNorm2d(bn_size * growth_rate)
        # relu2 = ReLU()
        # conv2 = Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
        raise NotImplementedError("TODO: Implement _DenseLayer")

    def forward(self, x):
        """Forward pass, returns concatenated features."""
        raise NotImplementedError("TODO: Implement forward")


class _DenseBlock(Module):
    """
    Dense block containing multiple dense layers.

    Args:
        num_layers: Number of dense layers
        num_input_features: Input channels to first layer
        bn_size: Bottleneck size
        growth_rate: Growth rate (k)
        drop_rate: Dropout rate
    """

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ):
        super().__init__()
        # TODO: Implement dense block
        raise NotImplementedError("TODO: Implement _DenseBlock")


class _Transition(Module):
    """
    Transition layer between dense blocks.
    Reduces feature maps and spatial dimensions.

    Args:
        num_input_features: Input channels
        num_output_features: Output channels
    """

    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        # TODO: Implement transition
        # bn = BatchNorm2d(num_input_features)
        # relu = ReLU()
        # conv = Conv2d(num_input_features, num_output_features, 1, bias=False)
        # pool = AvgPool2d(2, stride=2)
        raise NotImplementedError("TODO: Implement _Transition")


class DenseNet(Module):
    """
    DenseNet model.

    Args:
        growth_rate: Growth rate (k)
        block_config: Number of layers in each dense block
        num_init_features: Number of filters in initial convolution
        bn_size: Bottleneck multiplier
        drop_rate: Dropout rate
        num_classes: Number of output classes
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        num_classes: int = 1000,
    ):
        super().__init__()
        # TODO: Implement DenseNet
        raise NotImplementedError("TODO: Implement DenseNet")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def densenet121(num_classes: int = 1000, **kwargs) -> DenseNet:
    """DenseNet-121 model."""
    return DenseNet(32, (6, 12, 24, 16), 64, num_classes=num_classes, **kwargs)

def densenet161(num_classes: int = 1000, **kwargs) -> DenseNet:
    """DenseNet-161 model."""
    return DenseNet(48, (6, 12, 36, 24), 96, num_classes=num_classes, **kwargs)

def densenet169(num_classes: int = 1000, **kwargs) -> DenseNet:
    """DenseNet-169 model."""
    return DenseNet(32, (6, 12, 32, 32), 64, num_classes=num_classes, **kwargs)

def densenet201(num_classes: int = 1000, **kwargs) -> DenseNet:
    """DenseNet-201 model."""
    return DenseNet(32, (6, 12, 48, 32), 64, num_classes=num_classes, **kwargs)
