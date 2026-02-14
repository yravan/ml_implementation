"""
FCN - Fully Convolutional Networks
==================================

From "Fully Convolutional Networks for Semantic Segmentation"
https://arxiv.org/abs/1411.4038

Key innovations:
    1. Replace FC layers with 1x1 convolutions -> any input size
    2. Upsampling via transposed convolutions (deconvolution)
    3. Skip connections from earlier layers for fine details

FCN-32s: Direct 32x upsampling from conv5
FCN-16s: Combine conv5 (2x up) + conv4, then 16x up
FCN-8s: Combine conv5 + conv4 + conv3, then 8x up
"""

from typing import Optional, Dict
from python.nn_core import Module


class FCNHead(Module):
    """
    FCN head for semantic segmentation.

    Args:
        in_channels: Input channels from backbone
        channels: Intermediate channels
        num_classes: Number of output classes
    """

    def __init__(self, in_channels: int, channels: int, num_classes: int):
        super().__init__()
        # TODO: Implement FCN head
        # Conv2d(in_channels, channels, 3, padding=1)
        # BatchNorm2d(channels)
        # ReLU()
        # Dropout(0.1)
        # Conv2d(channels, num_classes, 1)
        raise NotImplementedError("TODO: Implement FCNHead")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


class FCN(Module):
    """
    Fully Convolutional Network for semantic segmentation.

    Args:
        backbone: Feature extraction backbone
        classifier: FCN head
        aux_classifier: Optional auxiliary classifier for deep supervision
    """

    def __init__(
        self,
        backbone: Module,
        classifier: Module,
        aux_classifier: Optional[Module] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

        # TODO: Implement FCN
        raise NotImplementedError("TODO: Implement FCN")

    def forward(self, x):
        """
        Forward pass.

        Returns:
            Dict with 'out' (main output) and optionally 'aux' (auxiliary output)
        """
        raise NotImplementedError("TODO: Implement forward")


def fcn_resnet50(num_classes: int = 21, aux_loss: bool = False, **kwargs) -> FCN:
    """FCN with ResNet-50 backbone."""
    raise NotImplementedError("TODO: Implement fcn_resnet50")


def fcn_resnet101(num_classes: int = 21, aux_loss: bool = False, **kwargs) -> FCN:
    """FCN with ResNet-101 backbone."""
    raise NotImplementedError("TODO: Implement fcn_resnet101")
