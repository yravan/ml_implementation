"""
Swin Transformer
================

Swin Transformer from "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
https://arxiv.org/abs/2103.14030

Key innovations:
    1. Hierarchical feature maps (like CNNs) - enables dense prediction tasks
    2. Shifted window attention - linear complexity w.r.t. image size
    3. Patch merging for downsampling

Window attention: Compute attention within local windows (e.g., 7x7)
Shifted windows: Alternate between regular and shifted window partitioning
    to enable cross-window connections.

Swin-V2 improvements:
    - Cosine attention (more stable at higher resolution)
    - Log-spaced continuous position bias
    - Residual post-normalization

Variants: Tiny, Small, Base, Large
"""

from typing import Optional, Tuple, List
from python.nn_core import Module


class PatchMerging(Module):
    """
    Patch Merging Layer.
    Reduces spatial resolution by 2x while increasing channels.

    Args:
        dim: Input dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        # TODO: Concatenate 2x2 patches and project
        # Reduces H, W by 2x, increases C by 2x (after projection)
        raise NotImplementedError("TODO: Implement PatchMerging")


class WindowAttention(Module):
    """
    Window-based Multi-head Self-Attention with relative position bias.

    Args:
        dim: Input dimension
        window_size: Window size (height, width)
        num_heads: Number of attention heads
        qkv_bias: Whether to use qkv bias
        attn_drop: Attention dropout
        proj_drop: Projection dropout
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement window attention with relative position bias
        raise NotImplementedError("TODO: Implement WindowAttention")

    def forward(self, x, mask: Optional = None):
        """
        Args:
            x: (num_windows*B, window_size*window_size, C)
            mask: Attention mask for shifted windows
        """
        raise NotImplementedError("TODO: Implement forward")


class SwinTransformerBlock(Module):
    """
    Swin Transformer Block.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        window_size: Window size
        shift_size: Shift size for shifted window attention
        mlp_ratio: MLP hidden dimension ratio
        qkv_bias: Whether to use qkv bias
        drop: Dropout rate
        attn_drop: Attention dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.shift_size = shift_size
        self.window_size = window_size
        # TODO: Implement Swin block
        # If shift_size > 0, need attention mask for cyclic shift
        raise NotImplementedError("TODO: Implement SwinTransformerBlock")

    def forward(self, x):
        """Forward with optional window shifting."""
        raise NotImplementedError("TODO: Implement forward")


class SwinTransformer(Module):
    """
    Swin Transformer.

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depths: Number of blocks in each stage
        num_heads: Number of attention heads in each stage
        window_size: Window size
        mlp_ratio: MLP ratio
        qkv_bias: Whether to use qkv bias
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement Swin Transformer
        raise NotImplementedError("TODO: Implement SwinTransformer")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


# Swin V1
def swin_t(num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """Swin-Tiny."""
    return SwinTransformer(
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        num_classes=num_classes, **kwargs
    )

def swin_s(num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """Swin-Small."""
    return SwinTransformer(
        embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
        num_classes=num_classes, **kwargs
    )

def swin_b(num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """Swin-Base."""
    return SwinTransformer(
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        num_classes=num_classes, **kwargs
    )

# Swin V2
def swin_v2_t(num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """Swin-V2-Tiny."""
    raise NotImplementedError("TODO: Implement swin_v2_t")

def swin_v2_s(num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """Swin-V2-Small."""
    raise NotImplementedError("TODO: Implement swin_v2_s")

def swin_v2_b(num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """Swin-V2-Base."""
    raise NotImplementedError("TODO: Implement swin_v2_b")
