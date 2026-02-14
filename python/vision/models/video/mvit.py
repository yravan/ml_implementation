"""
MViT - Multiscale Vision Transformer
=====================================

From "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
https://arxiv.org/abs/2112.01526

Key innovations:
    1. Multiscale feature hierarchy (like CNNs)
    2. Pooling attention for efficiency
    3. Decomposed relative position embeddings

MViT progressively pools spatial/temporal dimensions while expanding channels,
creating a feature pyramid within the transformer.
"""

from typing import List, Tuple
from python.nn_core import Module


class MultiscaleAttention(Module):
    """
    Multiscale attention with pooling.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV
        pool_kernel: Kernel for Q/K/V pooling
        pool_stride: Stride for pooling
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        pool_kernel: Tuple[int, int, int] = (3, 3, 3),
        pool_stride: Tuple[int, int, int] = (1, 1, 1),
    ):
        super().__init__()
        # TODO: Implement multiscale attention
        raise NotImplementedError("TODO: Implement MultiscaleAttention")


class MultiscaleBlock(Module):
    """MViT transformer block with optional pooling."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        pool_kernel: Tuple = (3, 3, 3),
        pool_stride: Tuple = (1, 1, 1),
    ):
        super().__init__()
        # TODO: Implement MViT block
        raise NotImplementedError("TODO: Implement MultiscaleBlock")


class MViT(Module):
    """
    Multiscale Vision Transformer.

    Args:
        spatial_size: Input spatial size (H, W)
        temporal_size: Input temporal size (T)
        in_chans: Input channels
        num_classes: Number of output classes
        embed_dim: Base embedding dimension
        depth: Total number of transformer blocks
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        spatial_size: int = 224,
        temporal_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 400,
        embed_dim: int = 96,
        depth: int = 16,
        num_heads: int = 1,
    ):
        super().__init__()
        # TODO: Implement MViT
        raise NotImplementedError("TODO: Implement MViT")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def mvit_v1_b(num_classes: int = 400, **kwargs) -> MViT:
    """MViT-V1-Base."""
    raise NotImplementedError("TODO: Implement mvit_v1_b")


def mvit_v2_s(num_classes: int = 400, **kwargs) -> MViT:
    """MViT-V2-Small."""
    raise NotImplementedError("TODO: Implement mvit_v2_s")
