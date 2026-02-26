"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

SimCLR demonstrates that contrastive learning with large batch sizes and
strong augmentations can achieve competitive performance with supervised learning.
Key insight: the method matters less than the quality of the contrastive objective.

Paper: "A Simple Framework for Contrastive Learning of Visual Representations"
       https://arxiv.org/abs/2002.05709
       Chen et al. (Google), 2020

Theory:
========
SimCLR (Simple Contrastive Learning of visual Representations) extends InfoNCE
with several key components that make it highly effective:

1. **Contrastive Objective**: InfoNCE loss with large batch sizes
   L = -log[exp(sim(z_i, z_i^+) / τ) / Σ_k exp(sim(z_i, z_k) / τ)]

2. **Large Batch Size**: N ≥ 256 (tested up to 4096)
   - More negatives → stronger gradient signal
   - Memory intensive but essential

3. **Strong Augmentations**: Random crop, color jitter, rotation
   - Creates diverse views of same sample
   - Prevents trivial solutions (collapse)

4. **Non-linear Projector**: 2-layer MLP projection head
   - Maps representations to projection space
   - Critical insight: improves representation quality
   - Architecture: z = h(f(x)) where h is MLP

5. **Symmetric Loss**: Uses both directions
   - L_total = L(z_i, z_i^+) + L(z_i^+, z_i)
   - Both samples contribute equally

Architecture:
==============

Input Image x
    ↓
[Augmentation] → x_i, x_i^+
    ↓
[Encoder f(·)] → h_i, h_i^+ ∈ ℝ^d  (d=2048, ResNet50)
    ↓
[Projection Head g(·)] → z_i, z_i^+ ∈ ℝ^c  (c=128, 2-layer MLP)
    ↓
[InfoNCE Loss]

Key Design Choices:
====================

1. Projection Head (CRITICAL):
   - Simple linear projection: POOR performance
   - 2-layer MLP (hidden 2048 → output 128): EXCELLENT
   - ReLU non-linearity between layers
   - NO normalization in projection head (normalize AFTER!)
   - Formula: z = g(h) where g(h) = W_2 ReLU(W_1 h + b_1) + b_2

   Why MLP works:
   - Non-linearity allows complex representations in h
   - Projection to lower dimension prevents information loss
   - ReLU non-linearity: sin(x), ReLU(x), tanh all work

2. Augmentation Pipeline:
   - Random crop (with resize)
   - Resize to original size
   - Horizontal flip
   - Color distortion (brightness, contrast, saturation, hue)
   - Gaussian blur
   - Grayscale conversion (10% probability)

   Ablation: Removing any single augmentation reduces performance
   Color distortion most important, followed by crop

3. Batch Normalization:
   - BatchNorm in encoder f(·): Essential for convergence
   - BatchNorm in projection head g(·): Yes, between layers
   - Synchronized BN across GPUs: Important for large batch sizes
   - Important: Don't use BN after final projection (before loss)

4. Temperature τ:
   - Value: 0.07 (found empirically)
   - Effect: Makes logits sharper, emphasizes hard negatives
   - Too high (τ > 0.5): Loss too soft, slow convergence
   - Too low (τ < 0.01): Numerical instability, gradients explode

5. Learning Rate:
   - Base LR: 0.3 (adjusted for batch size)
   - Formula: lr = 0.3 × (batch_size / 256)
   - Optimizer: SGD with momentum=0.9
   - Weight decay: 1e-6 (L2 regularization)

Training Procedure:
====================

For each iteration:
1. Sample batch of N images
2. Apply augmentation pipeline twice → 2N images (2 views per original)
3. Encode all 2N images through f(·) → 2N representations h ∈ ℝ^d
4. Project representations through g(·) → 2N representations z ∈ ℝ^c
5. Unit normalize z: z̄ = z / ||z||
6. Compute similarity matrix: logits = z̄ @ z̄^T / τ  [shape: 2N × 2N]
7. Create targets: image i's positive is image i' (the other view)
   - For first N images: positive indices are N to 2N-1
   - For next N images: positive indices are 0 to N-1
8. Compute loss: L = cross_entropy(logits, targets)
9. Backward and update encoder, projection head, batch norm params

Hyperparameter Ablations:
==========================

Effect of batch size (ImageNet):
  - Batch 256: 69.3% top-1 accuracy
  - Batch 512: 70.7%
  - Batch 1024: 71.3%
  - Batch 4096: 71.3% (diminishing returns)

Effect of projection head:
  - No projection: 60% accuracy
  - Linear projection: 68% accuracy
  - 2-layer MLP: 71% accuracy
  - 3-layer MLP: 71% accuracy (similar)

Effect of temperature:
  - τ = 0.05: 70.5% accuracy
  - τ = 0.07: 71.3% (optimal)
  - τ = 0.1: 71.1%
  - τ = 0.5: 65% (too soft)

Effect of augmentation:
  - With all augmentations: 71.3%
  - Without color distortion: 66%
  - Without crop: 65%
  - Without blur: 71%
  - Cropping most critical

Downstream Evaluation:
=======================

The learned representations are evaluated via:

1. **Linear Evaluation Protocol**:
   - Freeze encoder f(·)
   - Train only linear classifier on top
   - Measure accuracy: often 70-75% on ImageNet

2. **Transfer Learning**:
   - Fine-tune entire network on downstream task
   - Usually achieves supervised-like performance

3. **Semi-supervised Learning**:
   - Use pretrained features as initialization
   - Train with limited labeled data
   - Reduces label requirement significantly

Advanced Topics:
================

1. **Memory Consumption**:
   - Challenge: GPU memory needed for large batches
   - Solution: Gradient accumulation with sync every N steps
   - Alternative: Use memory bank (like MoCo) instead of in-batch negatives

2. **Multi-GPU Training**:
   - Requires synchronized batch normalization
   - All-gather operation to share representations across GPUs
   - Effective batch size = local_batch_size × num_gpus

3. **Preventing Collapse**:
   - Why it doesn't happen: Large batch size prevents trivial solutions
   - With small batches: representation collapse occurs (all similar)
   - Mitigation: Sufficient batch size (N ≥ 128)

4. **Representation Geometry**:
   - Representations form clusters by object category
   - Without labels, clusters form naturally
   - Projection head z = g(h) maps h to lower-dimensional sphere
"""

import numpy as np
from typing import Tuple, Optional, Callable
from python.nn_core import Module, Parameter


class AugmentationPipeline:
    """
    SimCLR's augmentation pipeline for creating positive pairs.

    Applies a sequence of random augmentations to generate two different
    views of the same image. Each call to __call__ produces two views.

    Key augmentations (in order of importance):
    1. Random crop and resize (prevents using full image features)
    2. Color distortion (brightness, contrast, saturation, hue)
    3. Gaussian blur (smooths local features)
    4. Grayscale (reduces color dependency)
    """

    def __init__(self, image_size: int = 224):
        """
        Args:
            image_size: Size of output image (typically 224 for ResNet)
        """
        raise NotImplementedError(
            "Implement augmentation pipeline:\n"
            "1. Random crop to 224x224 (or image_size)\n"
            "2. Random horizontal flip\n"
            "3. Color jitter (brightness, contrast, saturation, hue)\n"
            "4. Random Gaussian blur (kernel size 3 or 5)\n"
            "5. Random grayscale (10% probability)\n"
            "6. Normalize with ImageNet stats\n"
            "Hint: Use torchvision.transforms.Compose"
        )

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two augmented views of input image.

        Args:
            x: Input image tensor [3, H, W]

        Returns:
            Tuple of two augmented views
        """
        raise NotImplementedError(
            "Apply augmentation pipeline twice to create positive pair"
        )


class ProjectionHead(Module):
    """
    Non-linear projection head for SimCLR.

    Maps representations from encoder to projection space.
    Architecture: 2-layer MLP with ReLU non-linearity.

    Input: features from encoder f(x) ∈ ℝ^d
    Output: projections z ∈ ℝ^c (typically 128)

    Formula: z = g(h) = W_2 ReLU(BatchNorm(W_1 h))
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128):
        """
        Args:
            input_dim: Dimension of encoder output (e.g., 2048 for ResNet50)
            hidden_dim: Dimension of hidden layer (typically same as input_dim)
            output_dim: Dimension of projection (typically 128)
        """
        super().__init__()
        raise NotImplementedError(
            "Implement projection head:\n"
            "1. First linear layer: input_dim → hidden_dim\n"
            "2. BatchNorm1d on hidden_dim\n"
            "3. ReLU activation\n"
            "4. Second linear layer: hidden_dim → output_dim\n"
            "Formula: z = linear(ReLU(BatchNorm(linear(h))))"
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Project encoder features to embedding space.

        Args:
            h: Encoder features [batch_size, input_dim]

        Returns:
            Projections z [batch_size, output_dim]
        """
        raise NotImplementedError()


class SimCLRModel(Module):
    """
    Complete SimCLR model: encoder + projection head.

    Usage:
        model = SimCLRModel(encoder_name='resnet50')
        features_h = model.encoder(x)  # Get representations
        projections_z = model(x)  # Get projections for loss

    Architecture:
        Input Image → [Encoder f(·)] → h ∈ ℝ^d → [Projection g(·)] → z ∈ ℝ^c
    """

    def __init__(
        self,
        encoder_name: str = 'resnet50',
        projection_dim: int = 128,
        hidden_dim: int = 2048
    ):
        """
        Args:
            encoder_name: Name of encoder ('resnet50', 'resnet101', etc.)
            projection_dim: Dimension of projection output
            hidden_dim: Hidden dimension of projection head
        """
        super().__init__()
        raise NotImplementedError(
            "Implement SimCLR model:\n"
            "1. Load pretrained encoder (ResNet50 from torchvision)\n"
            "2. Remove classification head (keep backbone)\n"
            "3. Create ProjectionHead with appropriate dimensions\n"
            "Hint: Use torchvision.models.resnet50(pretrained=False)\n"
            "      then remove the fc layer"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get projections.

        Args:
            x: Input image [batch_size, 3, H, W]

        Returns:
            Projections z [batch_size, projection_dim]
        """
        raise NotImplementedError()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get encoder features without projection.

        Args:
            x: Input image [batch_size, 3, H, W]

        Returns:
            Features h [batch_size, 2048]
        """
        raise NotImplementedError()


class SimCLRLoss(Module):
    """
    Contrastive loss for SimCLR.

    Computes NT-Xent (Normalized Temperature-scaled Cross Entropy) loss:
    L = -log[exp(sim(z_i, z_i^+) / τ) / Σ_k exp(sim(z_i, z_k) / τ)]

    Key points:
    1. Takes normalized projections as input
    2. Uses cosine similarity (dot product after normalization)
    3. Temperature parameter τ controls loss sharpness
    4. Symmetric loss: average of both directions
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature for scaling similarity scores
                        Default 0.07 (empirically optimal)
        """
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for a batch of positive pairs.

        Args:
            z_i: Projections from first augmentation [batch_size, projection_dim]
            z_j: Projections from second augmentation [batch_size, projection_dim]
                 Assumed to be positive pairs with z_i (same order)

        Returns:
            Scalar loss value

        Implementation Steps:
        1. Concatenate z_i and z_j: z = [z_i; z_j] [2*batch_size, projection_dim]
        2. Normalize to unit sphere: z̄ = z / ||z||
        3. Compute similarity matrix: logits = z̄ @ z̄^T / τ [2*batch_size, 2*batch_size]
        4. Create labels where z_i's positive is z_j and vice versa
           - For indices 0..N-1 (z_i): positive is at indices N..2N-1 (z_j)
           - For indices N..2N-1 (z_j): positive is at indices 0..N-1 (z_i)
        5. Remove diagonal (self-similarity): mask out i's similarity to itself
        6. Apply cross-entropy loss: L = CrossEntropy(logits, labels)
        7. Symmetric: Return (L_i + L_j) / 2

        Hint: Use torch.cat, F.normalize, and the criterion
        """
        raise NotImplementedError(
            "Implement NT-Xent loss:\n"
            "1. Concatenate z_i and z_j along batch dimension\n"
            "2. Normalize concatenated tensor: z = F.normalize(z, dim=1)\n"
            "3. Compute logits: logits = (z @ z.T) / self.temperature\n"
            "4. Create labels: positions N..2N-1 for first N, 0..N-1 for last N\n"
            "5. Compute cross-entropy loss\n"
            "6. Optionally: compute symmetric loss (both directions)"
        )


class SimCLRTrainer:
    """
    Trainer for SimCLR self-supervised learning.

    Handles:
    - Loading data with augmentation pipeline
    - Training loop with gradient updates
    - Synchronized batch normalization for multi-GPU
    - Learning rate scheduling
    - Checkpoint saving/loading

    Usage:
        model = SimCLRModel()
        trainer = SimCLRTrainer(model, train_loader, device='cuda')
        for epoch in range(100):
            train_loss = trainer.train_epoch()
            print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
    """

    def __init__(
        self,
        model: SimCLRModel,
        optimizer,
        train_loader,
        loss_fn: SimCLRLoss,
        device: str = 'cpu',
        world_size: int = 1,
        rank: int = 0
    ):
        """
        Args:
            model: SimCLRModel instance
            optimizer: Optimizer (SGD with momentum recommended)
            train_loader: Training data loader
            loss_fn: SimCLRLoss instance
            device: 'cpu' (no GPU support in custom Module system)
            world_size: Number of GPUs (for distributed training)
            rank: Rank of current process (for distributed training)
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.device = device
        self.world_size = world_size
        self.rank = rank

    def train_epoch(self) -> float:
        """
        Train model for one epoch.

        Returns:
            Average training loss

        Training Loop:
        1. Set model to training mode
        2. For each batch:
           a. Load images: x [batch_size, 3, H, W]
           b. Create augmented pair: (x_i, x_j) from x
           c. Forward pass: z_i = model(x_i), z_j = model(x_j)
           d. Normalize projections to unit sphere
           e. Compute loss: L = loss_fn(z_i, z_j)
           f. Backward and optimizer step
           g. Track running loss
        3. Return average loss for epoch

        Important:
        - Use synchronized batch norm if multi-GPU
        - Scale learning rate by batch size
        - Log every N iterations
        """
        raise NotImplementedError(
            "Implement training loop:\n"
            "1. self.model.train()\n"
            "2. Iterate through train_loader\n"
            "3. For each batch, create two augmented views\n"
            "4. Encode both views: z_i, z_j = model(x_i), model(x_j)\n"
            "5. Normalize projections\n"
            "6. Compute loss and backprop\n"
            "7. Accumulate loss and return average"
        )

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint."""
        raise NotImplementedError(
            "Save model state_dict, optimizer state, and metadata to path"
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        raise NotImplementedError(
            "Load model state_dict and optimizer state from path"
        )


# ============================================================================
# Key Insights and Empirical Findings
# ============================================================================

"""
Why SimCLR Works So Well:

1. **Large Batch Size**:
   - In-batch negatives provide sufficient negative samples
   - Batch of 256+ prevents representation collapse
   - Enables unsupervised learning without additional memory bank

2. **Non-linear Projection Head**:
   - Simple linear projection performs poorly (68%)
   - 2-layer MLP jumps to 71% (3% absolute improvement)
   - ReLU non-linearity critical for capturing complexity
   - Learned representations even better than projections

3. **Strong Augmentations**:
   - Weak augmentations → poor representations
   - Strong augmentations create meaningful positive pairs
   - Prevents shortcut solutions (e.g., color matching)

4. **Temperature Parameter**:
   - Controls relative importance of hard vs easy negatives
   - τ = 0.07 found optimal through hyperparameter sweep
   - Sensitive hyperparameter but worth tuning per dataset

5. **Symmetry**:
   - Using both z_i→z_j and z_j→z_i improves performance
   - Each pair contributes equally to gradient

Common Mistakes:
- Using linear projection instead of MLP
- Forgetting batch normalization in projection head
- Too small batch size (< 128)
- Not normalizing projections before computing loss
- Using unsynchronized batch norm in multi-GPU setup

Extensions and Variants:
- SimCLRv2: Deeper projection head, larger model
- BYOL: Removes explicit negative samples (different paradigm)
- MoCo: Uses momentum encoder instead of large batches
"""


# Aliases for common naming conventions
SimCLR = SimCLRModel
NTXentLoss = SimCLRLoss
SimCLRAugmentation = AugmentationPipeline

