"""
Vision Transformer (ViT)
========================

Vision Transformer from "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
https://arxiv.org/abs/2010.11929

Key innovation: Apply standard Transformer architecture directly to images.
    1. Split image into fixed-size patches (e.g., 16x16)
    2. Linearly embed each patch
    3. Add position embeddings
    4. Feed sequence of vectors to standard Transformer encoder
    5. Use [CLS] token for classification

Notation: ViT-{size}/{patch_size}
    - ViT-B/16: Base model, 16x16 patches
    - ViT-L/32: Large model, 32x32 patches

Key insight: With sufficient data (JFT-300M), ViT outperforms CNNs
while being more computationally efficient at scale.
"""

from typing import Optional, Callable
from python.nn_core import Module


class PatchEmbed(Module):
    """
    Image to Patch Embedding.

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # TODO: Implement patch embedding
        # proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        raise NotImplementedError("TODO: Implement PatchEmbed")

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        raise NotImplementedError("TODO: Implement forward")


class Attention(Module):
    """
    Multi-head Self-Attention.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in qkv projection
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # TODO: Implement attention
        # qkv = Linear(dim, dim * 3, bias=qkv_bias)
        # attn_drop = Dropout(attn_drop)
        # proj = Linear(dim, dim)
        # proj_drop = Dropout(proj_drop)
        raise NotImplementedError("TODO: Implement Attention")

    def forward(self, x):
        """
        Args:
            x: (B, N, C)
        Returns:
            (B, N, C)
        """
        raise NotImplementedError("TODO: Implement forward")


class MLP(Module):
    """MLP block for Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement MLP
        # fc1 = Linear(in_features, hidden_features)
        # act = GELU()
        # drop1 = Dropout(drop)
        # fc2 = Linear(hidden_features, out_features)
        # drop2 = Dropout(drop)
        raise NotImplementedError("TODO: Implement MLP")


class TransformerBlock(Module):
    """
    Transformer encoder block.

    Structure:
        x -> LayerNorm -> Attention -> (+x) -> LayerNorm -> MLP -> (+x)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement transformer block
        raise NotImplementedError("TODO: Implement TransformerBlock")

    def forward(self, x):
        """Forward with pre-norm residual connections."""
        raise NotImplementedError("TODO: Implement forward")


class VisionTransformer(Module):
    """
    Vision Transformer.

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: Whether to use qkv bias
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement Vision Transformer
        # patch_embed = PatchEmbed(...)
        # cls_token = Parameter (1, 1, embed_dim)
        # pos_embed = Parameter (1, num_patches + 1, embed_dim)
        # pos_drop = Dropout(drop_rate)
        # blocks = [TransformerBlock(...) for _ in range(depth)]
        # norm = LayerNorm(embed_dim)
        # head = Linear(embed_dim, num_classes)
        raise NotImplementedError("TODO: Implement VisionTransformer")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


# Model constructors
def vit_b_16(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Base with 16x16 patches."""
    return VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        num_classes=num_classes, **kwargs
    )

def vit_b_32(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Base with 32x32 patches."""
    return VisionTransformer(
        patch_size=32, embed_dim=768, depth=12, num_heads=12,
        num_classes=num_classes, **kwargs
    )

def vit_l_16(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Large with 16x16 patches."""
    return VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        num_classes=num_classes, **kwargs
    )

def vit_l_32(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Large with 32x32 patches."""
    return VisionTransformer(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16,
        num_classes=num_classes, **kwargs
    )

def vit_h_14(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Huge with 14x14 patches."""
    return VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        num_classes=num_classes, **kwargs
    )
