"""
MaxViT
======

MaxViT from "MaxViT: Multi-Axis Vision Transformer"
https://arxiv.org/abs/2204.01697

Key innovation: Multi-axis attention that combines:
    1. Local (block) attention - like Swin Transformer
    2. Global (grid) attention - sparse global connectivity

This allows both local and global interactions while maintaining
linear complexity with respect to input size.

Architecture:
    - Hierarchical backbone (like Swin)
    - MaxViT blocks with: MBConv -> Block Attention -> Grid Attention
    - Block attention: Attention within non-overlapping windows
    - Grid attention: Attention across a dilated grid
"""

from typing import List, Optional, Tuple
from python.nn_core import Module


class MaxViTBlock(Module):
    """
    MaxViT block: MBConv + Block Attention + Grid Attention.

    Args:
        dim: Input dimension
        dim_out: Output dimension
        num_heads: Number of attention heads
        grid_size: Grid size for attention
        mlp_ratio: MLP ratio
        drop: Dropout rate
        attn_drop: Attention dropout rate
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        grid_size: Tuple[int, int] = (7, 7),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement MaxViT block
        # 1. MBConv block (local spatial mixing)
        # 2. Block attention (attention within windows)
        # 3. Grid attention (attention across dilated grid)
        raise NotImplementedError("TODO: Implement MaxViTBlock")

    def forward(self, x):
        """Forward through MBConv, block attention, and grid attention."""
        raise NotImplementedError("TODO: Implement forward")


class MaxViT(Module):
    """
    MaxViT model.

    Args:
        img_size: Input image size
        in_channels: Number of input channels
        num_classes: Number of output classes
        depths: Number of blocks in each stage
        channels: Channel dimensions for each stage
        num_heads: Number of attention heads in each stage
        grid_size: Grid size for attention
        mlp_ratio: MLP ratio
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 1000,
        depths: List[int] = [2, 2, 5, 2],
        channels: List[int] = [64, 128, 256, 512],
        num_heads: List[int] = [2, 4, 8, 16],
        grid_size: Tuple[int, int] = (7, 7),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement MaxViT
        raise NotImplementedError("TODO: Implement MaxViT")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def maxvit_t(num_classes: int = 1000, **kwargs) -> MaxViT:
    """MaxViT-Tiny."""
    return MaxViT(
        depths=[2, 2, 5, 2],
        channels=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        num_classes=num_classes,
        **kwargs
    )
