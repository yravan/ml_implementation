"""
Vision Transformer (ViT) Implementation

Module: sequence.transformers.vit

IMPLEMENTATION STATUS:
    - [ ] Image to patches conversion
    - [ ] Patch embedding and linear projection
    - [ ] Class token ([CLS]) and position embeddings
    - [ ] Transformer encoder stack
    - [ ] Classification head
    - [ ] Positional bias (optional)

COMPLEXITY:
    Time:  O(n^2 * d) where n = num_patches (typically 196 for 224x224 with 16x16 patches)
    Space: O(n * d) for storing activations
    Params: ~86M (ViT-Base), ~307M (ViT-Large), ~632M (ViT-Huge)

PREREQUISITES:
    - Understanding of transformer architecture
    - Knowledge of image processing (patches, embeddings)
    - Familiarity with CNN to vision transformer transition
    - PyTorch intermediate skills

REFERENCES:
    - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
      (Dosovitskiy et al., 2020) https://arxiv.org/abs/2010.11929
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - Google Brain Vision Transformer blog post

================================================================================
THEORY: Vision Transformer (ViT)
================================================================================

Vision Transformer represents a paradigm shift in computer vision:
from CNNs (convolutional inductive bias) to pure attention-based models.

KEY INSIGHTS AND DESIGN CHOICES:

1. CORE INNOVATION - PATCH EMBEDDING:
   - Divide image into fixed-size patches (e.g., 16x16 pixels)
   - Treat each patch as a "token" (similar to words in NLP)
   - Embed patches to d_model dimensions
   - Process with transformer (same as NLP)

   Example for 224x224 image with 16x16 patches:
   - Number of patches: (224/16)^2 = 14x14 = 196 patches
   - Each patch: 16x16x3 = 768 dimensions
   - Project to d_model: 196 x 768

2. NO INDUCTIVE BIAS (Unlike CNNs):
   - CNNs have: locality, translation equivariance, hierarchical structure
   - ViT has: none of these built-in
   - Must learn these properties from data
   - Requires large-scale pre-training (ImageNet-21k)
   - Once pre-trained, transfers well even with limited data

3. ADVANTAGES OVER CNNs:
   - Can process global context directly (attention is global)
   - Scalable to very large images (not limited by memory like convolutions)
   - More interpretable attention weights (can visualize what network attends to)
   - Same architecture for different image sizes (with position interpolation)
   - Better transfer learning with pre-training

4. CLASSIFICATION APPROACH:
   - Similar to BERT in NLP
   - Add learnable [CLS] token at the beginning
   - [CLS] token attends to all patches
   - Use [CLS] representation for classification (not global average pooling)

5. POSITION EMBEDDINGS:
   - Critical for transformers to understand spatial structure
   - ViT uses learnable 1D position embeddings
   - Not 2D grid embeddings (surprisingly, 1D works just as well)
   - During inference: can interpolate position embeddings for different image sizes

6. SCALING LAWS:
   - ViT-Tiny: 5.7M parameters
   - ViT-Small: 22M parameters
   - ViT-Base: 86M parameters
   - ViT-Large: 307M parameters
   - ViT-Huge: 632M parameters
   - Follows similar scaling laws as NLP transformers

7. HYBRID MODELS:
   - Can replace pure patch embedding with CNN backbone
   - CNN extracts features -> patches from CNN features
   - Combines inductive bias of CNNs with attention of transformers
   - Often better for small/medium datasets

================================================================================
MATHEMATICAL FORMULATION
================================================================================

PATCH EXTRACTION AND EMBEDDING:

    1. Divide image into non-overlapping patches
       For image H x W x C, patch size P x P:
       Number of patches: N = (H/P) * (W/P)

    2. Reshape and project
       Each patch: P x P x C -> P^2 * C dimensions
       Project to d_model: Linear(P^2 * C, d_model)

    3. Example (224x224x3 image, 16x16 patches):
       N = (224/16)^2 = 196 patches
       Patch dimension: 16^2 * 3 = 768
       Project: Linear(768, 768) for ViT-Base

SEQUENCE CONSTRUCTION:

    x_patches = Embed_Patches(image)  # [batch, num_patches, d_model]

    # Add class token at the beginning
    x_cls = Embed([CLS])  # [batch, 1, d_model]
    x = Concat(x_cls, x_patches)  # [batch, num_patches+1, d_model]

    # Add position embeddings
    positions = arange(num_patches + 1)
    x = x + Pos_Embed(positions)  # [batch, num_patches+1, d_model]

TRANSFORMER ENCODER:

    x = Dropout(x)

    For each transformer layer:
        x = x + MultiHeadAttention(LN(x))
        x = x + FFN(LN(x))

    x = LN(x)

CLASSIFICATION HEAD:

    clf_token = x[:, 0, :]  # Extract [CLS] token  # [batch, d_model]
    logits = Linear(d_model, num_classes)(clf_token)  # [batch, num_classes]

================================================================================
ARCHITECTURE OVERVIEW: Vision Transformer Stack
================================================================================

INPUT IMAGE: [batch, 3, 224, 224]
    |
    v
PATCH EMBEDDING:
    |-- Split into 196 patches (14x14 grid of 16x16 patches)
    |-- Linear projection: 768 -> 768 (or d_model)
    v
PATCH TOKENS: [batch, 196, 768]
    |
    v
ADD CLASS TOKEN: [batch, 197, 768]  (prepend [CLS] token)
    |
    v
ADD POSITION EMBEDDINGS: [batch, 197, 768]
    |
    v
EMBEDDING DROPOUT: [batch, 197, 768]
    |
    v
TRANSFORMER ENCODER LAYERS (12, 24, or 48):
    |
    +----> Layer 0:
    |       ├── LayerNorm
    |       ├── Multi-Head Self-Attention (full attention, no causal mask)
    |       ├── Residual
    |       ├── LayerNorm
    |       ├── MLP (FFN)
    |       └── Residual
    |
    +----> Layer 1-11 (same structure)
    |
    v
FINAL LAYER NORMALIZATION: [batch, 197, 768]
    |
    v
EXTRACT CLASS TOKEN: [batch, 768]  (use first token [CLS])
    |
    v
CLASSIFICATION HEAD: Linear(768, num_classes)
    |
    v
LOGITS: [batch, num_classes]

================================================================================
KEY DIFFERENCE: ViT vs CNN
================================================================================

ASPECT              CNN                         ViT
=================== ===========================  =======================
Receptive field     Local initially              Global from layer 1
Inductive bias      Locality, translation       None (learned from data)
Parameters          Fewer (50-100M)             More (86M-632M)
Pre-training        ImageNet (1.3M images)      ImageNet-21k (14M images)
Scaling             Limited (memory, compute)   Excellent (scales smoothly)
Interpretability    Hard (hidden features)      Easy (attention weights)
Transfer learning   Good                        Excellent
Architecture size   Different for 32x32 vs 224  Same (flexible input size)
Computation         Depends on image size       O(n^2 * d) where n = patches

TRADEOFFS:
- CNN: Fewer parameters, inductive bias, local efficiency
- ViT: More parameters, global context, better with scale, better transfer

================================================================================
FORWARD PASS SHAPE DOCUMENTATION
================================================================================

INPUT:
    x: [batch_size, channels, height, width] = [batch, 3, 224, 224]

AFTER PATCH EMBEDDING:
    patches: [batch, num_patches, patch_dim]
           = [batch, 196, 768]

AFTER ADDING [CLS] TOKEN:
    x: [batch, num_patches + 1, d_model]
     = [batch, 197, 768]

AFTER POSITION EMBEDDINGS:
    x: [batch, 197, 768]

AFTER EACH TRANSFORMER LAYER:
    x: [batch, 197, 768]

FINAL OUTPUT:
    x: [batch, 197, 768]

CLASSIFICATION OUTPUT:
    cls_token: [batch, 768]
    logits: [batch, num_classes]

================================================================================
COMMON VIT MODELS AND CONFIGURATIONS
================================================================================

ViT-Base (Dosovitskiy et al., 2020):
    d_model: 768, num_heads: 12, num_layers: 12, MLP dim: 3072
    Patch size: 16 (P=16)
    Parameters: 86M
    For 224x224 images: 196 patches

ViT-Large:
    d_model: 1024, num_heads: 16, num_layers: 24, MLP dim: 4096
    Patch size: 16
    Parameters: 307M

ViT-Huge:
    d_model: 1280, num_heads: 16, num_layers: 32, MLP dim: 5120
    Patch size: 14 (larger model can use smaller patches)
    Parameters: 632M

DeiT (Data Efficient Image Transformers):
    - Same architecture as ViT-Base
    - Better training strategies and regularization
    - Works well with ImageNet (1.3M) instead of ImageNet-21k (14M)

TIMM (PyTorch Image Models):
    - Popular library with many ViT variants
    - Includes hybrid models (CNN + ViT)
    - Pre-trained weights available

================================================================================
POSITION EMBEDDING INTERPOLATION
================================================================================

When using ViT on different image sizes:

1. Original image: 224x224 with 16x16 patches = 196 patches (14x14 grid)
2. New image: 384x384 with 16x16 patches = 576 patches (24x24 grid)

Problem: Position embeddings are learned for 196 patches
Solution: Interpolate position embeddings

Process:
    1. Reshape position embeddings to 2D: [14, 14, d_model]
    2. Bilinear interpolate to [24, 24, d_model]
    3. Flatten back to [576, d_model]
    4. Use interpolated embeddings

This allows ViT to process arbitrary image sizes at inference.

================================================================================
"""

import math
import numpy as np
from typing import Optional, Tuple, List

from python.nn_core import Module, Parameter, Sequential, ModuleList
from python.nn_core.layers.linear import Linear
from python.nn_core.normalization.layernorm import LayerNorm
from python.nn_core.attention.multihead import MultiHeadAttention
from python.nn_core.regularization.dropout import Dropout


class PatchEmbedding(Module):
    """
    Convert image to sequence of patch embeddings.

    Splits image into non-overlapping patches and projects each patch
    to a d_model-dimensional embedding.

    Args:
        img_size (int or tuple): Input image size (H, W). Default: 224
        patch_size (int): Patch size. Default: 16
        in_channels (int): Number of input channels. Default: 3 (RGB)
        embed_dim (int): Embedding dimension. Default: 768

    Shape:
        Input:  [batch_size, in_channels, height, width]
               e.g., [8, 3, 224, 224]
        Output: [batch_size, num_patches, embed_dim]
               e.g., [8, 196, 768] for 224x224 with 16x16 patches

    Computation:
        num_patches = (height // patch_size) * (width // patch_size)
        patch_dim = patch_size * patch_size * in_channels
        Linear(patch_dim, embed_dim)

    Example:
        >>> patch_embed = PatchEmbedding(img_size=224, patch_size=16)
        >>> x = np.random.randn(8, 3, 224, 224)
        >>> patches = patch_embed(x)
        >>> patches.shape
        Array shape([8, 196, 768])
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        """
        Initialize patch embedding.

        Args:
            img_size: Input image size
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        raise NotImplementedError(
            "PatchEmbedding.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Store img_size, patch_size, in_channels, embed_dim\n"
            "  2. Calculate num_patches = (img_size // patch_size) ** 2\n"
            "  3. Calculate patch_dim = patch_size * patch_size * in_channels\n"
            "  4. Create Conv2d(\n"
            "         in_channels, embed_dim,\n"
            "         kernel_size=patch_size,\n"
            "         stride=patch_size\n"
            "     ) to extract patches and project\n"
            "  5. Alternative: use Linear(patch_dim, embed_dim)\n"
            "     (requires manual patching)"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Convert image to patch embeddings.

        Args:
            x: [batch_size, in_channels, height, width]

        Returns:
            patches: [batch_size, num_patches, embed_dim]

        Process:
            1. Apply conv2d with kernel=patch_size, stride=patch_size
            2. Output shape: [batch, embed_dim, h_patches, w_patches]
            3. Flatten: [batch, embed_dim, h_patches*w_patches]
            4. Transpose: [batch, h_patches*w_patches, embed_dim]
        """
        raise NotImplementedError(
            "PatchEmbedding.forward not yet implemented.\n"
            "TODO:\n"
            "  1. # Using conv2d approach:\n"
            "     x = conv_proj(x)  # [batch, embed_dim, h_patches, w_patches]\n"
            "  2. # Flatten patches\n"
            "     x = x.flatten(2)  # [batch, embed_dim, num_patches]\n"
            "  3. # Transpose to [batch, num_patches, embed_dim]\n"
            "     x = x.transpose(1, 2)\n"
            "  4. return x"
        )


class VisionTransformer(Module):
    """
    Vision Transformer (ViT) for image classification.

    Treats image patches as tokens and applies transformer encoder.
    Suitable for:
    - Image classification
    - Fine-tuning for various downstream tasks
    - Transfer learning to small datasets (when pre-trained)

    Args:
        img_size (int): Input image size. Default: 224
        patch_size (int): Size of image patches. Default: 16
        in_channels (int): Number of input channels. Default: 3 (RGB)
        num_classes (int): Number of output classes. Default: 1000
        d_model (int): Model dimension. Default: 768 (ViT-Base)
        num_heads (int): Number of attention heads. Default: 12
        num_layers (int): Number of transformer layers. Default: 12
        d_ff (int): Feed-forward dimension. Default: 3072 (4*d_model)
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN ('relu' or 'gelu'). Default: 'gelu'
        use_cls_token (bool): Use class token for classification. Default: True
        pool_type (str): Pooling type ('cls' or 'mean'). Default: 'cls'

    Shape:
        Input:  [batch_size, 3, height, width]
        Output: [batch_size, num_classes]

    Example:
        >>> model = VisionTransformer(
        ...     img_size=224,
        ...     num_classes=1000,
        ...     d_model=768,
        ...     num_layers=12
        ... )
        >>> x = np.random.randn(8, 3, 224, 224)
        >>> logits = model(x)
        >>> logits.shape
        Array shape([8, 1000])

    Pre-training:
        Typically pre-trained on ImageNet-21k (14M images, 14k classes)
        Then fine-tuned on downstream tasks or ImageNet (1M images, 1k classes)

    References:
        "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
        (Dosovitskiy et al., 2020)
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
        use_cls_token: bool = True,
        pool_type: str = "cls",
    ):
        """
        Initialize Vision Transformer.

        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            num_classes: Number of output classes
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
            use_cls_token: Whether to use [CLS] token
            pool_type: Type of pooling ('cls' or 'mean')
        """
        super().__init__()
        raise NotImplementedError(
            "VisionTransformer.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Store all hyperparameters\n"
            "  2. Create PatchEmbedding to convert image to patches\n"
            "  3. Calculate num_patches\n"
            "  4. If use_cls_token: create learnable cls_token [1, 1, d_model]\n"
            "  5. Create positional embeddings:\n"
            "     Parameter with shape [num_patches + (1 if use_cls_token else 0), d_model]\n"
            "  6. Create embedding dropout\n"
            "  7. Create transformer encoder layers (ModuleList)\n"
            "  8. Create final layer norm\n"
            "  9. Create classification head: Linear(d_model, num_classes)\n"
            "  10. Store pool_type"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        ViT forward pass.

        Args:
            x: [batch_size, 3, height, width] - Input image

        Returns:
            logits: [batch_size, num_classes]

        Process:
            1. Convert image to patch embeddings
            2. Add class token (if enabled)
            3. Add positional embeddings
            4. Apply embedding dropout
            5. Pass through transformer encoder layers
            6. Apply final layer norm
            7. Pool (use [CLS] token or mean pooling)
            8. Classify

        Shape tracking:
            x: [batch, 3, 224, 224]
            -> [batch, 196, 768] after patch embedding
            -> [batch, 197, 768] after adding cls token
            -> [batch, 197, 768] after position embeddings + dropout
            -> [batch, 197, 768] after transformer
            -> [batch, 768] after pooling
            -> [batch, num_classes] after classification
        """
        raise NotImplementedError(
            "VisionTransformer.forward not yet implemented.\n"
            "TODO:\n"
            "  1. batch_size = x.size(0)\n"
            "  2. # Patch embedding\n"
            "     x = patch_embed(x)  # [batch, num_patches, d_model]\n"
            "  3. # Add class token\n"
            "     if use_cls_token:\n"
            "        cls = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]\n"
            "        x = torch.cat([cls, x], dim=1)  # [batch, num_patches+1, d_model]\n"
            "  4. # Add position embeddings\n"
            "     x = x + self.pos_embed\n"
            "  5. x = embedding_dropout(x)\n"
            "  6. # Transformer encoder\n"
            "     for layer in transformer_layers:\n"
            "        x = layer(x)\n"
            "  7. x = final_ln(x)\n"
            "  8. # Pooling\n"
            "     if pool_type == 'cls':\n"
            "        x = x[:, 0, :]  # [CLS] token\n"
            "     elif pool_type == 'mean':\n"
            "        x = x.mean(dim=1)  # mean pooling\n"
            "  9. logits = classification_head(x)\n"
            "  10. return logits"
        )

    def forward_features(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass returning features (before classification head).

        Args:
            x: [batch_size, 3, height, width]

        Returns:
            features: [batch_size, d_model]

        Useful for:
        - Fine-tuning on downstream tasks
        - Feature extraction
        - Attention visualization
        """
        raise NotImplementedError(
            "VisionTransformer.forward_features not yet implemented.\n"
            "TODO: Similar to forward but stop before classification head"
        )

    def get_attention_maps(self) -> List[np.ndarray]:
        """
        Extract attention maps from transformer layers.

        Returns:
            List of attention weight tensors from each layer.
            Each tensor shape: [batch_size, num_heads, num_patches+1, num_patches+1]

        Useful for:
            - Visualizing what the model attends to
            - Understanding spatial relationships learned
            - Probing transformer behavior
        """
        raise NotImplementedError(
            "VisionTransformer.get_attention_maps not yet implemented.\n"
            "TODO: Store and return attention weights from all layers"
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

# DeiT configurations (Data-Efficient Image Transformers)
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
