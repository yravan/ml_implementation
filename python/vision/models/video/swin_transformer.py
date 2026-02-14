"""
Video Swin Transformer
======================

From "Video Swin Transformer"
https://arxiv.org/abs/2106.13230

Extends Swin Transformer to video with 3D shifted windows.

Key adaptations:
    1. 3D patch embedding
    2. 3D window attention
    3. 3D shifted window mechanism
    4. 3D relative position bias
"""

from typing import List, Tuple, Optional
from python.nn_core import Module


class PatchEmbed3D(Module):
    """3D Patch Embedding for video."""

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        in_chans: int = 3,
        embed_dim: int = 96,
    ):
        super().__init__()
        # TODO: Conv3d for patch embedding
        raise NotImplementedError("TODO: Implement PatchEmbed3D")


class WindowAttention3D(Module):
    """3D Window attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement 3D window attention
        raise NotImplementedError("TODO: Implement WindowAttention3D")


class SwinTransformerBlock3D(Module):
    """Video Swin Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (8, 7, 7),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement 3D Swin block
        raise NotImplementedError("TODO: Implement SwinTransformerBlock3D")


class SwinTransformer3D(Module):
    """
    Video Swin Transformer.

    Args:
        patch_size: Patch size (T, H, W)
        in_chans: Input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depths: Number of blocks in each stage
        num_heads: Number of heads in each stage
        window_size: Window size
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        in_chans: int = 3,
        num_classes: int = 400,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: Tuple[int, int, int] = (8, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement Video Swin
        raise NotImplementedError("TODO: Implement SwinTransformer3D")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def swin3d_t(num_classes: int = 400, **kwargs) -> SwinTransformer3D:
    """Video Swin-Tiny."""
    return SwinTransformer3D(
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        num_classes=num_classes, **kwargs
    )


def swin3d_s(num_classes: int = 400, **kwargs) -> SwinTransformer3D:
    """Video Swin-Small."""
    return SwinTransformer3D(
        embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
        num_classes=num_classes, **kwargs
    )


def swin3d_b(num_classes: int = 400, **kwargs) -> SwinTransformer3D:
    """Video Swin-Base."""
    return SwinTransformer3D(
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        num_classes=num_classes, **kwargs
    )
