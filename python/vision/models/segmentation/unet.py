"""
UNet - Convolutional Networks for Biomedical Image Segmentation
===============================================================

From "U-Net: Convolutional Networks for Biomedical Image Segmentation"
https://arxiv.org/abs/1505.04597

Key innovations:
    1. Symmetric encoder-decoder with skip connections
    2. Skip connections concatenate encoder features with decoder features
    3. Works well with very few training images (data-efficient)
    4. Originally designed for biomedical segmentation

Architecture:
    Encoder (contracting path):
        [Conv3x3 -> BN -> ReLU] x2 -> MaxPool2x2  (repeated 4 times)

    Bottleneck:
        [Conv3x3 -> BN -> ReLU] x2

    Decoder (expanding path):
        Upsample2x -> Conv1x1 -> concat(skip) -> [Conv3x3 -> BN -> ReLU] x2

    Output:
        Conv1x1 -> num_classes

Channel progression (default):
    Encoder:  64 -> 128 -> 256 -> 512
    Bottleneck: 1024
    Decoder:  512 -> 256 -> 128 -> 64

Note: Input spatial dims should be divisible by 16 (4 pooling layers of stride 2).

Upsampling strategy:
    Use nearest-neighbor upsample + 1x1 conv instead of transposed convolution.
    This avoids checkerboard artifacts (see Odena et al., "Deconvolution and
    Checkerboard Artifacts", https://distill.pub/2016/deconv-checkerboard/).

References:
    - Original paper: https://arxiv.org/abs/1505.04597
    - 3D variant (V-Net): https://arxiv.org/abs/1606.04797
    - Attention U-Net: https://arxiv.org/abs/1804.03999
    - UNet++: https://arxiv.org/abs/1807.10165
"""

import numpy as np
from typing import List, Optional, Tuple
from python.nn_core import (
    Module,
    Sequential,
    ModuleList,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    Dropout2d,
    kaiming_normal_,
    zeros_,
)
from python.foundations import concat, Tensor
from python.foundations.functionals import Function
from python.foundations.computational_graph import convert_to_function


class NearestUpsample2dFunction(Function):
    """
    Nearest-neighbor 2x upsampling for 4D tensors (B, C, H, W).

    Forward: Repeat each spatial element 2x in both H and W.
        (B, C, H, W) -> (B, C, H*scale, W*scale) via np.repeat on axes 2,3

    Backward: Sum gradients from each scale x scale block back to original element.
        Reshape (B, C, H*s, W*s) -> (B, C, H, s, W, s) then sum over axes (3, 5)
    """

    def forward(self, x: np.ndarray, scale_factor: int = 2) -> np.ndarray:
        # TODO: Implement nearest-neighbor upsample forward
        # Save self.scale and self.input_shape for backward
        raise NotImplementedError("TODO: Implement NearestUpsample2dFunction forward")

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        # TODO: Implement nearest-neighbor upsample backward
        # Reshape grad to (B, C, H, s, W, s) and sum over (3, 5)
        raise NotImplementedError("TODO: Implement NearestUpsample2dFunction backward")


class Upsample2d(Module):
    """
    Nearest-neighbor 2x upsampling module.

    Doubles spatial dimensions: (B, C, H, W) -> (B, C, 2H, 2W).
    Fully differentiable with proper gradient support.

    Usage:
        self._upsample = convert_to_function(NearestUpsample2dFunction)
        # then in forward: self._upsample(x, scale_factor=self.scale_factor)

    Args:
        scale_factor: Integer upsampling factor (default: 2)
    """

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        # TODO: create self._upsample using convert_to_function
        raise NotImplementedError("TODO: Implement Upsample2d")

    def forward(self, x):
        raise NotImplementedError("TODO: Implement Upsample2d forward")


class DoubleConv(Module):
    """
    Double convolution block: (Conv3x3 -> BN -> ReLU) x 2.

    This is the fundamental building block of UNet, used in both
    the encoder and decoder paths.

    Structure:
        Conv2d(in, mid, 3, padding=1, bias=False) -> BN -> ReLU ->
        Conv2d(mid, out, 3, padding=1, bias=False) -> BN -> ReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        mid_channels: Intermediate channels (defaults to out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 mid_channels: Optional[int] = None):
        super().__init__()
        # TODO: Build self.double_conv as a Sequential of the layers above
        raise NotImplementedError("TODO: Implement DoubleConv")

    def forward(self, x):
        raise NotImplementedError("TODO: Implement DoubleConv forward")


class EncoderBlock(Module):
    """
    Encoder block: MaxPool -> DoubleConv.

    Downsamples spatial dimensions by 2x and increases channels.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # TODO: self.pool = MaxPool2d(kernel_size=2, stride=2)
        # TODO: self.conv = DoubleConv(in_channels, out_channels)
        raise NotImplementedError("TODO: Implement EncoderBlock")

    def forward(self, x):
        raise NotImplementedError("TODO: Implement EncoderBlock forward")


class DecoderBlock(Module):
    """
    Decoder block: Upsample2x -> Conv1x1 -> concat(skip) -> DoubleConv.

    Uses nearest-neighbor upsampling + 1x1 conv to halve channels,
    then concatenates with skip connection and applies double conv.

    Args:
        in_channels: Number of input channels (from lower level)
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # TODO: self.up = Upsample2d(scale_factor=2)
        # TODO: self.reduce = Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        # TODO: self.conv = DoubleConv(in_channels, out_channels)
        raise NotImplementedError("TODO: Implement DecoderBlock")

    def forward(self, x, skip):
        """
        Forward pass with skip connection.

        Args:
            x: Input from lower decoder level (B, C, H, W)
            skip: Skip connection from encoder (B, C/2, 2H, 2W)

        Returns:
            Decoded features (B, out_channels, 2H, 2W)

        Steps:
            x = self.up(x)        # (B, C, H, W) -> (B, C, 2H, 2W)
            x = self.reduce(x)    # (B, C, 2H, 2W) -> (B, C/2, 2H, 2W)
            x = concat(skip, x, axis=1)  # (B, C, 2H, 2W)
            return self.conv(x)
        """
        raise NotImplementedError("TODO: Implement DecoderBlock forward")


class UNet(Module):
    """
    UNet for semantic segmentation.

    Symmetric encoder-decoder architecture with skip connections.
    The encoder progressively downsamples while increasing channels,
    and the decoder upsamples while concatenating with encoder features.

    Args:
        in_channels: Number of input channels (e.g. 3 for RGB, 1 for grayscale)
        num_classes: Number of output segmentation classes
        features: Channel sizes for each encoder level.
            Default [64, 128, 256, 512] with bottleneck at 1024.
        dropout: Dropout probability applied after bottleneck (0 = no dropout)

    Example:
        >>> model = UNet(in_channels=3, num_classes=21)
        >>> x = Tensor(np.random.randn(2, 3, 256, 256))
        >>> out = model(x)  # shape: (2, 21, 256, 256)

    Input requirements:
        Spatial dimensions must be divisible by 2^(len(features)).
        With default features (4 levels), input H and W must be divisible by 16.

    Structure:
        self.inc = DoubleConv(in_channels, features[0])

        self.encoders = ModuleList of EncoderBlock for each level
            features[0]->features[1], features[1]->features[2], ...

        self.bottleneck = EncoderBlock(features[-1], features[-1]*2)

        self.decoders = ModuleList of DecoderBlock (reverse order)
            bottleneck->features[-1], features[-1]->features[-2], ...

        self.outc = Conv2d(features[0], num_classes, kernel_size=1)

    Forward:
        1. x = inc(x), save skip
        2. For each encoder: x = encoder(x), save skip
        3. x = bottleneck(x), optional dropout
        4. For each decoder: x = decoder(x, skip) using skips in reverse
        5. return outc(x)

    Weight init: kaiming_normal_ for conv weights, zeros_ for biases
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        features: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement UNet __init__
        raise NotImplementedError("TODO: Implement UNet")

    def _init_weights(self):
        """Initialize weights with Kaiming normal for conv layers."""
        raise NotImplementedError("TODO: Implement _init_weights")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Output tensor of shape (B, num_classes, H, W)
        """
        raise NotImplementedError("TODO: Implement UNet forward")


class UNetSmall(Module):
    """
    Smaller UNet variant: features = [32, 64, 128, 256].

    ~7.8M parameters. Good for smaller datasets.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 21,
                 dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError("TODO: Implement UNetSmall")

    def forward(self, x):
        raise NotImplementedError("TODO: Implement UNetSmall forward")


class UNetTiny(Module):
    """
    Tiny UNet: features = [16, 32, 64] (3 levels).

    ~0.5M parameters. Input spatial dims must be divisible by 8.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 21):
        super().__init__()
        raise NotImplementedError("TODO: Implement UNetTiny")

    def forward(self, x):
        raise NotImplementedError("TODO: Implement UNetTiny forward")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def unet(in_channels: int = 3, num_classes: int = 21, **kwargs) -> UNet:
    """Standard UNet with default channel progression [64, 128, 256, 512]. ~31M params."""
    return UNet(in_channels=in_channels, num_classes=num_classes, **kwargs)


def unet_small(in_channels: int = 3, num_classes: int = 21, **kwargs) -> UNetSmall:
    """Small UNet with channel progression [32, 64, 128, 256]. ~7.8M params."""
    return UNetSmall(in_channels=in_channels, num_classes=num_classes, **kwargs)


def unet_tiny(in_channels: int = 3, num_classes: int = 21) -> UNetTiny:
    """Tiny UNet with channel progression [16, 32, 64]. ~0.5M params. Divisible by 8."""
    return UNetTiny(in_channels=in_channels, num_classes=num_classes)
