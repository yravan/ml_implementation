"""
VGG
===

VGG model architectures from "Very Deep Convolutional Networks for Large-Scale
Image Recognition" - https://arxiv.org/abs/1409.1556

Key insight: Deep networks with small (3x3) filters outperform shallow networks
with larger filters, while being more computationally efficient.

Configurations:
    VGG11: 8 conv + 3 FC = 11 weight layers
    VGG13: 10 conv + 3 FC = 13 weight layers
    VGG16: 13 conv + 3 FC = 16 weight layers
    VGG19: 16 conv + 3 FC = 19 weight layers

Each variant also has a batch normalization version (*_bn).

Architecture pattern:
    - Stack of 3x3 conv layers (same padding)
    - MaxPool 2x2, stride 2 after each block
    - Three FC layers: 4096 -> 4096 -> num_classes
    - ReLU activation throughout
"""

from typing import List, Union, Optional
from python.nn_core import (
    Module,
    Sequential,
    AdaptiveAvgPool2d,
    ReLU,
    Dropout,
    Linear,
    Conv2d,
    MaxPool2d,
)
from python.nn_core.module import Flatten

# VGG configurations
# Numbers = conv output channels, 'M' = MaxPool
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG11
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG13
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG16
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],  # VGG19
}


class VGG(Module):
    """
    VGG model.

    Args:
        features: Feature extraction layers (conv + pool)
        num_classes: Number of output classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        features: Module,
        num_classes: int = 1000,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.features = features
        self.num_classes = num_classes

        self.classifier = Sequential(
            AdaptiveAvgPool2d((7, 7)),
            Flatten(),
            Linear(512 * 7 * 7, 4096), ReLU(), Dropout(dropout),
            Linear(4096, 4096), ReLU(), Dropout(dropout),
            Linear(4096, num_classes)
        )


    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[int, str]], batch_norm: bool = False) -> Module:
    """
    Create VGG feature layers from config.

    Args:
        cfg: Configuration list (channel counts and 'M' for maxpool)
        batch_norm: Whether to use batch normalization

    Returns:
        Sequential module of feature layers
    """
    feature_layers = []
    prev_channels = 3
    for channels in cfg:
        if isinstance(channels, int):
            feature_layers.append(Conv2d(prev_channels, channels, kernel_size=3, padding=1))
            feature_layers.append(ReLU())
        elif isinstance(channels, str) and channels == 'M':
            feature_layers.append(MaxPool2d(kernel_size=2, stride=2))
        else:
            raise ValueError("Unknown channel number {}".format(channels))
    return Sequential(*feature_layers)


def _vgg(cfg: str, batch_norm: bool, num_classes: int, **kwargs) -> VGG:
    """Create a VGG model."""
    features = make_layers(cfgs[cfg], batch_norm=batch_norm)
    return VGG(features, num_classes=num_classes, **kwargs)


# Model constructors
def vgg11(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-11 model."""
    return _vgg('A', False, num_classes, **kwargs)

def vgg11_bn(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-11 model with batch normalization."""
    return _vgg('A', True, num_classes, **kwargs)

def vgg13(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-13 model."""
    return _vgg('B', False, num_classes, **kwargs)

def vgg13_bn(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-13 model with batch normalization."""
    return _vgg('B', True, num_classes, **kwargs)

def vgg16(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-16 model."""
    return _vgg('D', False, num_classes, **kwargs)

def vgg16_bn(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-16 model with batch normalization."""
    return _vgg('D', True, num_classes, **kwargs)

def vgg19(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-19 model."""
    return _vgg('E', False, num_classes, **kwargs)

def vgg19_bn(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-19 model with batch normalization."""
    return _vgg('E', True, num_classes, **kwargs)
