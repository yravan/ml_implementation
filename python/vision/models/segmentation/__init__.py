"""
Semantic Segmentation Models
============================

Models for pixel-wise classification.

Available models:
- UNet: Encoder-decoder with skip connections
- FCN: Fully Convolutional Networks
- DeepLabV3: Atrous Spatial Pyramid Pooling
- LRASPP: Lite R-ASPP for mobile
"""

from .unet import (
    unet, unet_small, unet_tiny,
    UNet, UNetSmall, UNetTiny,
    DoubleConv, EncoderBlock, DecoderBlock,
)
from .fcn import fcn_resnet50, fcn_resnet101, FCN
from .deeplabv3 import (
    deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large,
    DeepLabV3, DeepLabHead, ASPP
)
from .lraspp import lraspp_mobilenet_v3_large, LRASPP

__all__ = [
    'unet', 'unet_small', 'unet_tiny',
    'UNet', 'UNetSmall', 'UNetTiny',
    'DoubleConv', 'EncoderBlock', 'DecoderBlock',
    'fcn_resnet50', 'fcn_resnet101', 'FCN',
    'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large',
    'DeepLabV3', 'DeepLabHead', 'ASPP',
    'lraspp_mobilenet_v3_large', 'LRASPP',
]
