"""
ConvNeXt
========

ConvNeXt architecture from "A ConvNet for the 2020s"
https://arxiv.org/abs/2201.03545

Key insight: Modernize ResNet with techniques from Vision Transformers.

Modifications from ResNet:
    1. Training: Similar to ViT (AdamW, augmentations, longer training)
    2. Macro design: Stage compute ratio 3:3:9:3 (like Swin)
    3. Stem: Patchify with 4x4 non-overlapping conv
    4. ResNeXt-ify: Depthwise convolution
    5. Inverted bottleneck: Wide -> narrow -> wide
    6. Large kernel: 7x7 depthwise convs
    7. Micro design: GELU, fewer activations/norms, LayerNorm, separate downsampling

The result: A pure ConvNet that matches Swin Transformer performance!
"""

from typing import List, Optional, Callable
from python.nn_core import Module


class LayerNorm2d(Module):
    """LayerNorm for 2D inputs (channels first)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        # TODO: Implement LayerNorm for (N, C, H, W) tensors
        raise NotImplementedError("TODO: Implement LayerNorm2d")


class CNBlock(Module):
    """
    ConvNeXt block.

    Structure:
        DWConv 7x7 -> LayerNorm -> Linear (expand) -> GELU -> Linear (project)
        With residual connection.

    Args:
        dim: Number of channels
        drop_path: Stochastic depth rate
        layer_scale_init_value: Initial value for layer scale
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        # TODO: Implement ConvNeXt block
        # dwconv = Conv2d(dim, dim, 7, padding=3, groups=dim)
        # norm = LayerNorm(dim)
        # pwconv1 = Linear(dim, 4 * dim)  # Expansion
        # act = GELU()
        # pwconv2 = Linear(4 * dim, dim)  # Projection
        # layer_scale (learnable parameter)
        raise NotImplementedError("TODO: Implement CNBlock")

    def forward(self, x):
        """Forward with residual."""
        raise NotImplementedError("TODO: Implement forward")


class ConvNeXt(Module):
    """
    ConvNeXt model.

    Args:
        in_chans: Number of input channels
        num_classes: Number of output classes
        depths: Number of blocks at each stage
        dims: Feature dimensions at each stage
        drop_path_rate: Stochastic depth rate
        layer_scale_init_value: Init value for layer scale
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        # TODO: Implement ConvNeXt
        # Stem: Conv2d(in_chans, dims[0], 4, stride=4) + LayerNorm
        # Stages: Each stage has depth[i] blocks + downsampling layer
        # Head: Global avg pool + LayerNorm + Linear
        raise NotImplementedError("TODO: Implement ConvNeXt")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def convnext_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    """ConvNeXt-Tiny."""
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes, **kwargs)

def convnext_small(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    """ConvNeXt-Small."""
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], num_classes=num_classes, **kwargs)

def convnext_base(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    """ConvNeXt-Base."""
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=num_classes, **kwargs)

def convnext_large(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    """ConvNeXt-Large."""
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], num_classes=num_classes, **kwargs)
