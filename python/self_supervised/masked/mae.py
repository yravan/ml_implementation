"""
MAE: Masked Autoencoders Are Scalable Vision Learners

Extends masked language modeling (MLM) from NLP to vision by masking random
patches of images and training to reconstruct them. MAE shows that Vision
Transformers can learn effective representations through simple reconstruction.

Paper: "Masked Autoencoders Are Scalable Vision Learners"
       https://arxiv.org/abs/2111.06377
       He et al. (Meta AI Research), 2021

Theory:
========
Key Insight: Reconstruction as Self-Supervised Learning for Vision

While masked language modeling (MLM) works well for NLP, vision has unique
properties that enable different approaches:

1. **Information Redundancy**:
   - Images have high spatial redundancy
   - Nearby pixels/patches are correlated
   - Can reconstruct unmasked regions from visible patches
   - Unlike text where each token is distinct

2. **Asymmetric Efficiency**:
   - Encoder: Process only visible patches (~25% of 196 patches)
   - Decoder: Process all patches (including masked)
   - Encoder is small, decoder is small
   - Total compute much less than processing full image

3. **Learned Representation Through Reconstruction**:
   - Encoder learns to extract meaningful features
   - Decoder learns to reason about spatial relationships
   - Both emerge naturally from reconstruction objective

Architecture Overview:
======================

Vision Transformer (ViT) based architecture:

Input Image (224×224)
       ↓
Divide into patches (16×16 patches) → 196 patches
       ↓
Patch Embedding + Positional Embedding
       ↓
Random Masking (75% masks, keep 25%)
       ↓
[Encoder (ViT Blocks)] → Processes visible patches only (~49 patches)
       ↓
Encoder Output: Encoded features of visible patches
       ↓
Add Mask Tokens + Positional Embeddings
       ↓
[Decoder (ViT Blocks)] → Processes all patch positions (196)
       ↓
Prediction Head (Linear)
       ↓
Reconstructed Patch Values (196 patches × 3 × 256 values per channel)

Key Insight: Encoder only processes 25% of patches, yet learns good representations!

Masking Strategy:
=================

MAE uses Random Masking:
  - 75% of patches randomly selected to mask
  - 25% of patches remain visible
  - No "fake" patch values or special tokens
  - Simply remove masked patches from input

Why 75%?
  - Very high masking ratio (unusual in NLP where 15% is common)
  - Can work because vision has high redundancy
  - Forces encoder to learn discriminative features
  - Patches must be inferred from distant context

Comparison to MLM (NLP):

  NLP Masking:
    - 15% of tokens masked
    - Each token is distinct
    - Replacing with [MASK] token provides signal

  MAE (Vision):
    - 75% of patches masked
    - Patches are visually redundant
    - No replacement token, just remove masked patches
    - Encoder doesn't see masked regions at all

This asymmetry enables efficiency: encoder is lightweight because it only sees
visible patches.

Encoder-Decoder Architecture:
============================

Encoder:
  - ViT encoder blocks
  - Processes only visible patches (25% of sequence)
  - Small, efficient
  - Output: Encoded features of visible patches

Decoder:
  - ViT decoder blocks (typically smaller than encoder)
  - Processes all patches:
    - Encoded features at visible positions
    - Mask tokens at masked positions
  - Each mask token learnable parameter
  - Generates prediction for every patch position

Mask Tokens:
  - Learnable token (unlike [MASK] in BERT)
  - Same embedding for all masked positions (not tied to position)
  - Added to positional embeddings

Prediction Head:
  - Simple linear projection
  - Input: decoder output for each patch position
  - Output: pixel values for reconstruction
  - Shape: [196, 3*16*16] for 224×224 image with 16×16 patches

Loss Function:
===============

Reconstruction loss using pixel values:

L = MSE(reconstructed_patches, original_patches)

But only computed on masked patches:

L = ||y_masked - ŷ_masked||_2^2

Where:
  - y_masked: Original pixel values of masked patches
  - ŷ_masked: Reconstructed pixel values from decoder
  - MSE computed only over 75% masked patches

Why only masked patches?
  - Decoder has full visibility of unmasked patches
  - So decoding unmasked patches is trivial
  - Learning signal comes from reconstructing hidden patches

Normalization:
  - Patches normalized before reconstruction
  - Means and stds computed per channel
  - Helps with loss scaling (patches naturally normalized)

Training Procedure:
====================

1. Load image (224×224)

2. Divide into patches (16×16) → 196 patches
   Shape: [196, 3*256] = [196, 768] after linearization

3. Randomly mask 75% → keep 25% (~49 patches visible)

4. Encode visible patches:
   - Add positional embeddings (for visible positions only)
   - Pass through encoder blocks
   - Output shape: [49, encoder_dim]

5. Prepare decoder input:
   - Add mask tokens for masked positions
   - Add positional embeddings (full positions)
   - Concatenate: [196, decoder_dim]

6. Decode:
   - Pass through decoder blocks
   - Output shape: [196, decoder_dim]

7. Predict:
   - Linear head: decoder_dim → patch_size*patch_size*3
   - Output shape: [196, 768] for pixel values

8. Compute loss:
   - Extract reconstructions for masked patches
   - Compare to original masked patches
   - Compute MSE loss

9. Backprop and update all parameters

Why MAE Works:
===============

1. **Semantic Learning**: Even though reconstructing pixels seems low-level,
   the learned representations are semantic (capture objects, scenes, etc.)

2. **Efficient Processing**: By masking 75%, encoder only processes 25% of
   computation, leading to 4× efficiency gain. Yet representations as good
   as processing full image.

3. **Scaling**: Unlike supervised learning, MAE improves with scale.
   Larger models learn better representations.

4. **Transfer Learning**: Learned representations transfer well to downstream
   tasks despite learning via pixel reconstruction.

Why Representations are Good:
  - To reconstruct masked patches, must understand objects
  - Model learns texture, shape, color, spatial relationships
  - These are exactly what's needed for downstream vision tasks

Downstream Evaluation:
======================

1. **Linear Probing**:
   - Freeze MAE encoder
   - Train linear classifier on top of encoder features
   - ImageNet accuracy: ~75-80%

2. **Fine-tuning**:
   - Entire encoder updated on downstream task
   - Better performance than linear probing
   - Usually ~82-84% on ImageNet

3. **Transfer Learning**:
   - Works well on diverse tasks (detection, segmentation)
   - Better initialization than supervised ImageNet

Scaling Properties:
===================

Effect of Model Scale (ImageNet fine-tuning):

  ViT-Base:   81-82% (224×224)
  ViT-Large:  82-83%
  ViT-Huge:   83-84%

Scaling to larger models consistently improves performance!

This is different from supervised learning where scaling has diminishing returns.

Effect of Masking Ratio:

  25% masked:   73% accuracy (linear eval)
  50% masked:   77%
  75% masked:   80% (optimal)
  90% masked:   79% (slightly worse)

75% masking is empirically optimal for vision.

Comparison with Other Methods:
==============================

                    MAE         CLIP        SimCLR      Supervised
Paradigm:          Reconstruction Contrastive Contrastive Supervised
Input Modality:    Image only  Image+Text  Image       Image+Labels
Fine-tune Acc:     83-84%      ~82%       69%*        82%
Transfer to Other: Excellent   Excellent  Good        Good
Scalability:       Excellent   Limited    Fair        Limited

*SimCLR on ViT (not original ResNet)

MAE shines in:
  - Scalability (larger = better)
  - Transfer learning
  - Data efficiency (good with fewer labels)

Advanced Topics:
================

1. **Asymmetric Masking Ratio**:
   MAE uses different ratios for different patches:
     - Content patches (foreground): lower masking rate
     - Background patches: higher masking rate
   Can be done adaptively or with region-aware masking

2. **Patch Size Trade-offs**:

   Patch Size 8×8:
     - More patches (784)
     - Finer granularity
     - Harder reconstruction (smaller patches)

   Patch Size 16×16:
     - Standard for ViT (196 patches)
     - Good balance

   Patch Size 32×32:
     - Coarser (49 patches)
     - Easier reconstruction
     - Less semantic information

3. **Decoder Asymmetry**:
   Decoder typically smaller than encoder:
     - Encoder: 12 blocks
     - Decoder: 8 blocks
   Saves computation while maintaining performance

4. **Position-Aware Masking**:
   Can vary masking ratio by position:
     - Center: Lower masking ratio (more informative)
     - Edges: Higher masking ratio (more redundant)
   Not standard but researched in variants

5. **Hierarchical Masking**:
   Mask at multiple levels:
     - Coarse: Mask entire regions
     - Fine: Mask individual patches
   Creates hierarchical learning

Implementation Considerations:
============================

1. **Efficiency Tricks**:
   - Encoder input: Only visible patches (remove masked)
   - Saves memory and computation
   - Must restore positions before decoder

2. **Memory Requirements**:
   - Encoder: Small (processes 49 patches)
   - Decoder: Medium (processes 196 patches)
   - Overall: Smaller than processing full image

3. **Positional Embeddings**:
   - Must handle sparse positions in encoder
   - Decoder needs full positional embeddings
   - Important for position awareness

4. **Reconstruction Target**:
   - Pixels vs. features vs. tokens
   - Paper uses pixels (simplest)
   - Features or tokens might be more semantic
   - Pixels work surprisingly well

5. **Loss Weighting**:
   - Can weight patches by importance
   - Foreground patches: Higher weight
   - Background patches: Lower weight
   - Optional but can improve performance

Variants and Extensions:
=======================

1. **CAN (Channel Attention Networks)**:
   - Attention to channel importance
   - Weight reconstruction by informative channels

2. **BEiT (BERT pre-training for Image Transformers)**:
   - Discrete tokens instead of pixels
   - Vectorized Discrete Autoencoder
   - More like BERT approach to vision

3. **SimMIM**:
   - Simple approach similar to MAE
   - Direct pixel reconstruction
   - Comparable performance

4. **OpenAI MAE (Continued Learning)**:
   - Pre-train on diverse image datasets
   - Shows scaling laws for vision

Why Masking Rather than Contrastive?
===================================

1. **Computational Efficiency**:
   - MAE: Process 25% of patches
   - Contrastive: Process 100% or large batches
   - MAE much more efficient

2. **Scalability**:
   - MAE improves with model size
   - Contrastive (SimCLR) needs large batches
   - MAE easier to scale

3. **Data Efficiency**:
   - MAE works well with less data
   - Contrastive needs more negatives

4. **Natural for Vision**:
   - Images have high redundancy
   - Masking exploits this naturally
   - Contrastive doesn't leverage spatial structure

Theoretical Understanding:
===========================

Information-Theoretic View:
  - Image has spatial structure/redundancy
  - Masking 75% removes pixels but information remains
  - Decoder recovers via spatial reasoning
  - Efficient information bottleneck

Connection to Compression:
  - Similar to lossy image compression
  - Encoder: Compress to latent
  - Decoder: Decompress (reconstruct)
  - But with noisy compression (masking)

Why Reconstruction Learns Semantics:
  - Low-level: Requires understanding texture, edges
  - Mid-level: Requires object parts, shapes
  - High-level: Requires scene understanding
  - All emerge from reconstruction objective

Practical Recommendations:
==========================

For Implementation:

1. Use Vision Transformer backbone
2. Mask 75% of patches randomly
3. Encoder: Process visible patches only
4. Decoder: Add mask tokens + positional embeddings
5. Reconstruct pixel values of masked patches
6. Use MSE loss on masked patches only

Key Hyperparameters:

  - Masking ratio: 75% (critical!)
  - Patch size: 16×16 (standard)
  - Encoder blocks: 12
  - Decoder blocks: 8
  - Hidden dim: 768 (ViT-Base)
  - Learning rate: 1.5e-4
  - Batch size: 512
  - Epochs: 400-800 (long training helps)

Results Expectations:

  - Linear evaluation: 75-80% on ImageNet
  - Fine-tuning: 82-84%
  - Transfer to other tasks: Very good
  - Scales better than contrastive methods
"""

import numpy as np
from typing import Tuple, Optional, List
from python.nn_core import Module, Parameter


class PatchEmbedding(Module):
    """
    Convert image to patch embeddings.

    Divides image into non-overlapping patches and creates embeddings.

    Example:
      Image: [3, 224, 224]
      Patches: 16×16 → 196 patches
      Each patch: 3 × 256 values (768-dim)

    Args:
        img_size: Size of input image (assumed square)
        patch_size: Size of each patch (e.g., 16)
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Size of patches to extract
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        raise NotImplementedError(
            "Implement patch embedding:\n"
            "1. Create conv2d layer:\n"
            "   - kernel_size = patch_size\n"
            "   - stride = patch_size\n"
            "   - output channels = embed_dim\n"
            "2. This extracts patches and projects to embed_dim\n"
            "Formula: Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Extract and embed patches.

        Args:
            x: Input image [batch_size, 3, H, W]

        Returns:
            Patch embeddings [batch_size, num_patches, embed_dim]

        Implementation:
        1. Apply convolution to extract patches
        2. Reshape to [batch_size, num_patches, embed_dim]
        3. Return embeddings
        """
        raise NotImplementedError()


class RandomPatchMasking:
    """
    Randomly mask patches for MAE training.

    Args:
        masking_ratio: Fraction of patches to mask (typically 0.75)
    """

    def __init__(self, masking_ratio: float = 0.75):
        """
        Args:
            masking_ratio: Fraction to mask (default 0.75 = 75%)
        """
        self.masking_ratio = masking_ratio

    def __call__(
        self,
        patch_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomly mask patches and return visible patches.

        Args:
            patch_embeddings: [batch_size, num_patches, embed_dim]

        Returns:
            visible_patches: Embeddings of visible (unmasked) patches
            mask: Binary mask [batch_size, num_patches]
            restore_order: Indices to restore original order

        Implementation:
        1. Get number of patches N
        2. Compute number to keep: N_keep = int(N * (1 - masking_ratio))
        3. Random permutation: perm = randperm(N)
        4. Visible indices: first N_keep from permutation
        5. Create mask and return
        """
        raise NotImplementedError(
            "Implement masking:\n"
            "1. num_patches = patch_embeddings.shape[1]\n"
            "2. keep_ratio = 1.0 - self.masking_ratio\n"
            "3. num_keep = int(num_patches * keep_ratio)\n"
            "4. perm = torch.randperm(num_patches)\n"
            "5. visible_idx = perm[:num_keep]\n"
            "6. mask: all zeros except visible positions\n"
            "7. Return visible_embeddings, mask, perm"
        )


class MAEEncoder(Module):
    """
    Encoder for Masked Autoencoder.

    Processes only visible patches.

    Typically smaller than in other ViT applications because
    it only processes 25% of patches.

    Architecture:
      Visible Patches → Positional Embedding → ViT Blocks → Output
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer = None
    ):
        """
        Args:
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            norm_layer: Normalization layer
        """
        super().__init__()
        raise NotImplementedError(
            "Implement encoder:\n"
            "1. Create stack of Transformer blocks\n"
            "2. Store embed_dim for later use\n"
            "Note: Encoder typically has 12 blocks"
        )

    def forward(self, x: np.ndarray, pos_embed: np.ndarray) -> np.ndarray:
        """
        Encode visible patches.

        Args:
            x: Patch embeddings [batch_size, num_visible, embed_dim]
            pos_embed: Positional embeddings for visible positions

        Returns:
            Encoded features [batch_size, num_visible, embed_dim]
        """
        raise NotImplementedError()


class MAEDecoder(Module):
    """
    Decoder for Masked Autoencoder.

    Processes all patch positions (visible + masked).

    Args:
        embed_dim: Embedding dimension
        decoder_embed_dim: Embedding dimension in decoder (usually same)
        decoder_depth: Number of decoder blocks
        decoder_num_heads: Number of attention heads
        mlp_ratio: MLP ratio
        norm_layer: Normalization layer
    """

    def __init__(
        self,
        embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer = None
    ):
        """
        Args:
            embed_dim: Encoder embedding dimension
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder blocks (typically 8)
            decoder_num_heads: Number of attention heads
            mlp_ratio: MLP ratio
            norm_layer: Normalization layer
        """
        super().__init__()
        raise NotImplementedError(
            "Implement decoder:\n"
            "1. Create projection: embed_dim → decoder_embed_dim\n"
            "2. Create learnable mask tokens\n"
            "3. Create stack of Transformer blocks\n"
            "4. Store dimensions for later use\n"
            "Note: Decoder typically has 8 blocks"
        )

    def forward(
        self,
        x: np.ndarray,
        pos_embed: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Decode to reconstruct all patches.

        Args:
            x: Encoded features [batch_size, num_visible, embed_dim]
            pos_embed: Positional embeddings [batch_size, num_patches, decoder_embed_dim]
            mask: Mask indicating visible/masked positions [batch_size, num_patches]

        Returns:
            Decoded features [batch_size, num_patches, decoder_embed_dim]

        Implementation:
        1. Project x to decoder_embed_dim
        2. Insert masked tokens at masked positions
        3. Add positional embeddings
        4. Pass through decoder blocks
        5. Return decoded features
        """
        raise NotImplementedError()


class MAEPredictionHead(Module):
    """
    Prediction head to reconstruct patch values from decoder output.

    Maps decoder output to pixel values.

    Args:
        decoder_embed_dim: Dimension of decoder output
        patch_size: Size of each patch (16)
        in_channels: Number of channels (3 for RGB)
    """

    def __init__(
        self,
        decoder_embed_dim: int = 512,
        patch_size: int = 16,
        in_channels: int = 3
    ):
        """
        Args:
            decoder_embed_dim: Decoder embedding dimension
            patch_size: Size of patches
            in_channels: Number of input channels (RGB = 3)
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.output_dim = patch_size * patch_size * in_channels

        raise NotImplementedError(
            "Implement prediction head:\n"
            "1. Linear layer: decoder_embed_dim → patch_size*patch_size*in_channels\n"
            "Formula: Linear(decoder_embed_dim, output_dim)"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Reconstruct patch values.

        Args:
            x: Decoder output [batch_size, num_patches, decoder_embed_dim]

        Returns:
            Reconstructed patches [batch_size, num_patches, output_dim]
        """
        raise NotImplementedError()


class MAEModel(Module):
    """
    Complete Masked Autoencoder model.

    Combines encoder, decoder, and prediction head.

    Usage:
        model = MAEModel()
        loss = model.train_loss(images)
        loss.backward()
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        masking_ratio: float = 0.75
    ):
        """
        Args:
            img_size: Input image size (assumed square)
            patch_size: Size of patches
            in_channels: Input channels (3 for RGB)
            embed_dim: Encoder embedding dimension
            encoder_depth: Number of encoder blocks
            encoder_num_heads: Number of encoder attention heads
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder blocks
            decoder_num_heads: Number of decoder attention heads
            masking_ratio: Ratio of patches to mask (typically 0.75)
        """
        super().__init__()
        raise NotImplementedError(
            "Implement MAE model:\n"
            "1. Patch embedding layer\n"
            "2. Positional embeddings (learnable)\n"
            "3. Encoder\n"
            "4. Decoder\n"
            "5. Prediction head\n"
            "6. Masking strategy\n"
            "7. Store configuration"
        )

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass to get reconstruction.

        Args:
            x: Input image [batch_size, 3, H, W]

        Returns:
            pred: Reconstructed patches [batch_size, num_patches, patch_dim]
            mask: Binary mask [batch_size, num_patches]
        """
        raise NotImplementedError(
            "Implement forward:\n"
            "1. Extract patches: patch_emb = embed(x)\n"
            "2. Mask patches: vis_patches, mask, restore_idx = masking(patch_emb)\n"
            "3. Encode: encoded = encoder(vis_patches)\n"
            "4. Decode: decoded = decoder(encoded, mask)\n"
            "5. Predict: pred = head(decoded)\n"
            "6. Return pred, mask"
        )


class MAELoss(Module):
    """
    Reconstruction loss for MAE.

    Computes MSE loss only on masked patches.

    L = MSE(reconstructed_masked, original_masked)
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Compute reconstruction loss on masked patches only.

        Args:
            pred: Predicted patch values [batch_size, num_patches, patch_dim]
            target: Original patch values [batch_size, num_patches, patch_dim]
            mask: Binary mask [batch_size, num_patches], 1 = masked

        Returns:
            Scalar loss value

        Implementation:
        1. Compute MSE: mse = (pred - target)^2
        2. Average over patch dimensions: mse_per_patch = mse.mean(dim=-1)
        3. Apply mask: masked_loss = mse_per_patch * mask
        4. Average over batch: loss = masked_loss.sum() / mask.sum()
        5. Return loss
        """
        raise NotImplementedError(
            "Implement MAE loss:\n"
            "1. mse = (pred - target) ** 2  # [batch, num_patches, patch_dim]\n"
            "2. mse = mse.mean(dim=-1)  # Average over patch dimension\n"
            "3. loss = (mse * mask).sum() / (mask.sum() + 1e-6)\n"
            "4. Return loss\n"
            "Note: Only masked patches contribute to loss"
        )


class MAETrainer:
    """
    Trainer for Masked Autoencoder.

    Handles:
    - Loading image data
    - Training loop with masking
    - Reconstruction loss computation
    - Checkpoint saving/loading

    Usage:
        model = MAEModel()
        trainer = MAETrainer(model, train_loader, device='cuda')
        for epoch in range(400):
            train_loss = trainer.train_epoch()
    """

    def __init__(
        self,
        model: MAEModel,
        optimizer,
        train_loader,
        loss_fn: MAELoss,
        device: str = 'cpu',
        val_loader = None
    ):
        """
        Args:
            model: MAEModel instance
            optimizer: Optimizer (AdamW recommended)
            train_loader: Training data loader
            loss_fn: MAELoss instance
            device: 'cpu' (no GPU support in custom Module system)
            val_loader: Optional validation data loader
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss

        Training Loop:
        1. Set model to training mode
        2. For each batch of images:
           a. Forward pass: pred, mask = model(images)
           b. Create target by extracting patch values from images
           c. Compute loss: loss = loss_fn(pred, target, mask)
           d. Backward and optimizer step
           e. Track running loss
        3. Return average loss

        Important:
        - Target should be patches with same normalization as input
        - Loss only computed on masked patches
        - Careful with normalizations (mean/std)
        """
        raise NotImplementedError(
            "Implement training loop:\n"
            "1. self.model.train()\n"
            "2. For each batch:\n"
            "   a. images = batch.to(self.device)\n"
            "   b. pred, mask = self.model(images)\n"
            "   c. target = extract_patches(images)\n"
            "   d. loss = self.loss_fn(pred, target, mask)\n"
            "   e. loss.backward()\n"
            "   f. self.optimizer.step()\n"
            "   g. Track loss\n"
            "3. Return average loss"
        )

    def evaluate(self) -> float:
        """Evaluate on validation set."""
        raise NotImplementedError()

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint."""
        raise NotImplementedError()

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        raise NotImplementedError()


# ============================================================================
# Understanding MAE
# ============================================================================

"""
Key Insights about Masked Autoencoders for Vision:

1. **Why Masking Works for Vision**:
   - Images have spatial redundancy
   - Nearby regions correlated
   - Can infer masked regions from context
   - Unlike language where each token distinct

2. **Asymmetric Design is Key**:
   - Encoder: Processes 25% (only visible patches)
   - Decoder: Processes 100%
   - Enables 4× compute savings during pretraining
   - Yet learn representations as good as processing all

3. **High Masking Ratio**:
   - MAE: 75% masking (very aggressive)
   - MLM (NLP): 15% masking
   - Difference: Vision has more redundancy
   - Aggressive masking forces learning of structure

4. **Scaling Properties**:
   - Larger models learn better with MAE
   - Different from supervised learning (plateau)
   - Shows self-supervised learning scales better

5. **Simple Reconstruction**:
   - Predicting pixels seems "low-level"
   - Yet learns semantic representations
   - Because understanding scenes requires semantic knowledge

Future Directions:
  - Multi-scale masking
  - Adaptive masking ratios
  - Combining with other losses
  - Applications to other modalities (3D, video)
"""
