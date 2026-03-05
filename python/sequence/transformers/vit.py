"""
Vision Transformer (ViT) Implementation

Module: sequence.transformers.vit

COMPLEXITY:
    Time:  O(n^2 * d) where n = num_patches (typically 196 for 224x224 with 16x16 patches)
    Space: O(n * d) for storing activations
    Params: ~86M (ViT-Base), ~307M (ViT-Large), ~632M (ViT-Huge)

REFERENCES:
    - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
      (Dosovitskiy et al., 2020) https://arxiv.org/abs/2010.11929
    - "Attention Is All You Need" (Vaswani et al., 2017)

================================================================================
THEORY: Vision Transformer (ViT)
================================================================================

Vision Transformer represents a paradigm shift in computer vision:
from CNNs (convolutional inductive bias) to pure attention-based models.

KEY INSIGHTS AND DESIGN CHOICES:

1. CORE INNOVATION - PATCH EMBEDDING:
   - Divide image into fixed-size patches (e.g., 16x16 pixels)
   - Treat each patch as a "token" (similar to words in NLP)
   - Embed patches to d_model dimensions via Conv2d or linear projection

   Example for 224x224 image with 16x16 patches:
   - Number of patches: (224/16)^2 = 196
   - Each patch: 16x16x3 = 768 dimensions

2. NO INDUCTIVE BIAS (Unlike CNNs):
   - Must learn locality and translation equivariance from data
   - Requires large-scale pre-training (ImageNet-21k)

3. CLASSIFICATION APPROACH:
   - Add learnable [CLS] token at the beginning
   - Use [CLS] representation for classification

4. POSITION EMBEDDINGS:
   - Learnable 1D position embeddings (not 2D grid)
   - Can interpolate for different image sizes at inference

================================================================================
MATHEMATICAL FORMULATION
================================================================================

PATCH EXTRACTION AND EMBEDDING:
    patches = Conv2d(in_channels, d_model, kernel=patch_size, stride=patch_size)
    x_patches: [batch, num_patches, d_model]

SEQUENCE CONSTRUCTION:
    x = Concat([CLS], x_patches) + positional_embeddings
    x: [batch, num_patches + 1, d_model]

TRANSFORMER ENCODER:
    For each layer: x = x + MHA(LN(x)); x = x + FFN(LN(x))
    x = LN(x)

CLASSIFICATION:
    logits = Linear(x[:, 0, :])  # [CLS] token -> num_classes

================================================================================
"""

import numpy as np
from typing import Optional, List

from python.foundations import Tensor
from python.nn_core import Module, Parameter, ModuleList
from python.nn_core.linear import Linear
from python.nn_core.normalization import LayerNorm
from python.nn_core.attention import MultiHeadAttention
from python.nn_core.regularization import Dropout
from python.nn_core.activations import GELU
from python.nn_core.conv import Conv2d


class PatchEmbedding(Module):
    """
    Convert image to sequence of patch embeddings.

    Splits an image into a grid of non-overlapping patches and embeds
    each patch as a token using a Conv2d with kernel_size=stride=patch_size.

    Args:
        img_size (int): Input image size. Default: 224
        patch_size (int): Patch size. Default: 16
        in_channels (int): Number of input channels. Default: 3
        embed_dim (int): Embedding dimension. Default: 768

    Shape:
        Input:  [batch_size, in_channels, height, width]
        Output: [batch_size, num_patches, embed_dim]
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        raise NotImplementedError(
            "Patch embedding uses a 2D convolution with kernel size and stride "
            "equal to the patch size to simultaneously extract and project "
            "non-overlapping image patches into the model's embedding dimension. "
            "The convolution output is then reshaped from spatial dimensions "
            "into a sequence of patch tokens."
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Convert image to patch embeddings.

        Args:
            x: [batch_size, in_channels, height, width]

        Returns:
            [batch_size, num_patches, embed_dim]
        """
        raise NotImplementedError(
            "Applies the convolutional projection to extract patches, "
            "flattens the spatial dimensions into a sequence dimension, "
            "and transposes to produce a sequence of patch embeddings."
        )


class VisionTransformer(Module):
    """
    Vision Transformer (ViT) for image classification.

    Applies a standard transformer encoder to image patch sequences.
    Splits the input image into patches, embeds them, prepends a learnable
    [CLS] token, adds learned positional embeddings, and processes through
    N transformer encoder layers before classifying via the [CLS] token.

    Args:
        img_size (int): Input image size. Default: 224
        patch_size (int): Patch size. Default: 16
        in_channels (int): Number of input channels. Default: 3
        num_classes (int): Number of output classes. Default: 1000
        d_model (int): Model dimension. Default: 768
        num_heads (int): Number of attention heads. Default: 12
        num_layers (int): Number of transformer layers. Default: 12
        d_ff (int): Feed-forward dimension. Default: 3072
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN. Default: 'gelu'
        pool_type (str): Pooling type ('cls' or 'mean'). Default: 'cls'

    Shape:
        Input:  [batch_size, in_channels, height, width]
        Output: [batch_size, num_classes]
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        pool_type: str = "cls",
    ):
        super().__init__()
        raise NotImplementedError(
            "The Vision Transformer uses a patch embedding layer (Conv2d-based) "
            "to convert the image into a sequence of patch tokens, prepends a "
            "learnable [CLS] token, adds learned positional embeddings for all "
            "positions (patches + CLS), applies dropout, and processes through "
            "N transformer encoder layers with bidirectional self-attention. "
            "A final layer normalization is applied, and the [CLS] token "
            "(or mean-pooled representation) is projected through a linear "
            "classification head."
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        ViT forward pass.

        Args:
            x: [batch_size, in_channels, height, width]

        Returns:
            logits: [batch_size, num_classes]
        """
        raise NotImplementedError(
            "Converts the image to patch embeddings, prepends the [CLS] token, "
            "adds positional embeddings, applies dropout, passes through the "
            "transformer encoder stack, applies final normalization, pools "
            "the sequence representation (via [CLS] or mean pooling), and "
            "projects through the classification head."
        )

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Forward pass returning features before classification head.

        Args:
            x: [batch_size, in_channels, height, width]

        Returns:
            features: [batch_size, d_model]
        """
        raise NotImplementedError(
            "Same as forward but stops before the classification head, "
            "returning the pooled feature representation. Useful for "
            "fine-tuning, feature extraction, and transfer learning."
        )


# ============================================================================
# CONFIGURATION DICTIONARIES FOR COMMON VIT MODELS
# ============================================================================

VIT_BASE_CONFIG = {
    "img_size": 224,
    "patch_size": 16,
    "in_channels": 3,
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 12,
    "d_ff": 3072,
    "dropout": 0.1,
    "activation": "gelu",
}

VIT_LARGE_CONFIG = {
    "img_size": 224,
    "patch_size": 16,
    "in_channels": 3,
    "d_model": 1024,
    "num_heads": 16,
    "num_layers": 24,
    "d_ff": 4096,
    "dropout": 0.1,
    "activation": "gelu",
}

VIT_HUGE_CONFIG = {
    "img_size": 224,
    "patch_size": 14,
    "in_channels": 3,
    "d_model": 1280,
    "num_heads": 16,
    "num_layers": 32,
    "d_ff": 5120,
    "dropout": 0.1,
    "activation": "gelu",
}

DEIT_BASE_CONFIG = {
    "img_size": 224,
    "patch_size": 16,
    "in_channels": 3,
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 12,
    "d_ff": 3072,
    "dropout": 0.1,
    "activation": "gelu",
}

DEIT_SMALL_CONFIG = {
    "img_size": 224,
    "patch_size": 16,
    "in_channels": 3,
    "d_model": 384,
    "num_heads": 6,
    "num_layers": 12,
    "d_ff": 1536,
    "dropout": 0.1,
    "activation": "gelu",
}
