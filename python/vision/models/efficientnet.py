"""
EfficientNet
============

EfficientNet architectures from:
- "EfficientNet: Rethinking Model Scaling for CNNs" https://arxiv.org/abs/1905.11946
- "EfficientNetV2: Smaller Models and Faster Training" https://arxiv.org/abs/2104.00298

Key innovation (V1): Compound scaling
    - Scale depth, width, and resolution together with fixed ratio
    - depth = alpha^phi, width = beta^phi, resolution = gamma^phi
    - alpha * beta^2 * gamma^2 ≈ 2 (compute constraint)

EfficientNet-B0 is the baseline, B1-B7 are scaled versions.

Key improvements (V2):
    - Fused-MBConv in early layers (faster training)
    - Progressive learning (gradually increase image size)
    - Adaptive regularization
"""

from typing import List, Optional, Tuple, Callable
from python.nn_core import Module


class MBConvConfig:
    """Configuration for MBConv blocks."""

    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
    ):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self._adjust_channels(input_channels, width_mult)
        self.out_channels = self._adjust_channels(out_channels, width_mult)
        self.num_layers = self._adjust_depth(num_layers, depth_mult)

    @staticmethod
    def _adjust_channels(channels: int, width_mult: float) -> int:
        return int(channels * width_mult)

    @staticmethod
    def _adjust_depth(num_layers: int, depth_mult: float) -> int:
        return int(max(1, num_layers * depth_mult))


class MBConv(Module):
    """
    Mobile Inverted Bottleneck Conv (MBConv) block.

    Args:
        config: MBConvConfig
        stochastic_depth_prob: Drop probability for stochastic depth
    """

    def __init__(self, config: MBConvConfig, stochastic_depth_prob: float = 0.0):
        super().__init__()
        # TODO: Implement MBConv
        # 1. Expansion phase: 1x1 conv (if expand_ratio != 1)
        # 2. Depthwise conv: kernel x kernel
        # 3. Squeeze-and-Excitation
        # 4. Projection phase: 1x1 conv
        # 5. Skip connection (if stride == 1 and in_channels == out_channels)
        raise NotImplementedError("TODO: Implement MBConv")


class FusedMBConv(Module):
    """
    Fused MBConv for EfficientNetV2.
    Replaces depthwise + 1x1 with single 3x3 conv in early layers.
    """

    def __init__(self, config: MBConvConfig, stochastic_depth_prob: float = 0.0):
        super().__init__()
        # TODO: Implement FusedMBConv
        raise NotImplementedError("TODO: Implement FusedMBConv")


class EfficientNet(Module):
    """
    EfficientNet model.

    Args:
        inverted_residual_setting: Architecture config
        dropout: Dropout probability
        stochastic_depth_prob: Stochastic depth probability
        num_classes: Number of output classes
    """

    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float = 0.2,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
    ):
        super().__init__()
        # TODO: Implement EfficientNet
        raise NotImplementedError("TODO: Implement EfficientNet")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def _efficientnet(
    width_mult: float,
    depth_mult: float,
    dropout: float,
    num_classes: int,
    **kwargs
) -> EfficientNet:
    """Create EfficientNet with given scaling."""
    # TODO: Define base architecture and scale
    raise NotImplementedError("TODO: Implement _efficientnet")


# EfficientNet B0-B7
def efficientnet_b0(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """EfficientNet-B0 (baseline)."""
    return _efficientnet(1.0, 1.0, 0.2, num_classes, **kwargs)

def efficientnet_b1(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """EfficientNet-B1."""
    return _efficientnet(1.0, 1.1, 0.2, num_classes, **kwargs)

def efficientnet_b2(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """EfficientNet-B2."""
    return _efficientnet(1.1, 1.2, 0.3, num_classes, **kwargs)

def efficientnet_b3(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """EfficientNet-B3."""
    return _efficientnet(1.2, 1.4, 0.3, num_classes, **kwargs)

def efficientnet_b4(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """EfficientNet-B4."""
    return _efficientnet(1.4, 1.8, 0.4, num_classes, **kwargs)

def efficientnet_b5(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """EfficientNet-B5."""
    return _efficientnet(1.6, 2.2, 0.4, num_classes, **kwargs)

def efficientnet_b6(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """EfficientNet-B6."""
    return _efficientnet(1.8, 2.6, 0.5, num_classes, **kwargs)

def efficientnet_b7(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """EfficientNet-B7."""
    return _efficientnet(2.0, 3.1, 0.5, num_classes, **kwargs)

# EfficientNet V2
def efficientnet_v2_s(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """EfficientNet V2 Small."""
    raise NotImplementedError("TODO: Implement efficientnet_v2_s")

def efficientnet_v2_m(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """EfficientNet V2 Medium."""
    raise NotImplementedError("TODO: Implement efficientnet_v2_m")

def efficientnet_v2_l(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """EfficientNet V2 Large."""
    raise NotImplementedError("TODO: Implement efficientnet_v2_l")
