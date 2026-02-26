"""
InfoNCE Loss: Information Noise-Contrastive Estimation

A foundational contrastive learning objective that maximizes mutual information
between encoded representations from different views of the same sample.

Paper: "Representation Learning with Contrastive Predictive Coding"
       https://arxiv.org/abs/1807.03748
       van den Oord et al., 2018

Theory:
========
InfoNCE (Information Noise-Contrastive Estimation) is a self-supervised learning
objective that learns representations by contrasting a positive pair against
negative samples from the same batch.

Given:
  - An anchor sample x_i
  - A positive sample x_i^+ (different augmentation of same x_i)
  - N-1 negative samples {x_j^-} from other samples in the batch

The InfoNCE loss measures how well we can distinguish the positive from negatives:

    L_InfoNCE = -log[exp(sim(z_i, z_i^+) / τ) /
                      Σ_k exp(sim(z_i, z_k) / τ)]

Where:
  - z_i, z_i^+ = encoded representations
  - sim(·,·) = similarity metric (cosine similarity or dot product)
  - τ = temperature parameter (controls sharpness)
  - The denominator includes positive + all negatives in batch

Key Properties:
  1. Instance discrimination: Each sample-augmentation pair is its own class
  2. Symmetry: Can optimize from both directions (i→i^+ and i^+→i)
  3. Batch size dependency: More negatives → stronger gradients
  4. Temperature sensitivity: τ ∈ [0.1, 1.0] typically

Mathematical Details:
====================
The loss can be derived from information theory. It lower-bounds the mutual
information I(z_i; z_i^+):

    I(z_i; z_i^+) ≥ log(N) - L_InfoNCE

This means minimizing the loss directly maximizes mutual information between
the two views, subject to having N negative samples.

Contrastive Learning Interpretation:
  The loss pulls together positive pairs (numerator)
  while pushing apart negative samples (denominator).

  Think of it as: "How confidently can I pick the positive from negatives?"

Temperature Effects:
  - τ → 0: Loss becomes sharper, focuses on hardest negatives
  - τ → ∞: Loss becomes softer, all negatives treated equally
  - Typical: τ ≈ 0.07 (SimCLR), 0.1 (MoCo)

Training Procedure:
===================
1. Sample mini-batch of N samples
2. Apply two different augmentations to each sample → 2N representations
3. Encode all 2N samples → get embeddings z
4. Normalize z to unit sphere (important!)
5. Compute similarity matrix: S[i,j] = z_i^T z_j
6. For each sample i:
   - Compute loss using i as anchor, i' (same sample, diff augment) as positive
   - All other 2(N-1) samples are negatives
7. Backprop and update encoder

Practical Considerations:
=========================
- Memory Bank: Original CPC used memory bank to have more negatives
- Projector Head: Use MLP projection head (2-layer) before computing loss
- Normalization: Unit normalization essential for cosine similarity
- Batch Size: Need sufficient N for stable training (N ≥ 256)
- Hard Negatives: Harder negatives produce stronger learning signal

Advantages:
  + Simple and elegant formulation
  + Theoretically grounded in information theory
  + Foundation for modern contrastive methods
  + Works with small feature dimensions

Disadvantages:
  - Requires large batch size for negative samples
  - All batch samples act as negatives (may include similar samples)
  - Computational cost grows with batch size
"""

import numpy as np
from typing import Tuple, Optional
from python.nn_core import Module, Parameter


class InfoNCELoss(Module):
    """
    InfoNCE Loss Implementation

    Computes the Information Noise-Contrastive Estimation loss for a batch
    of samples with positive pairs (same sample, different augmentations).

    Args:
        temperature (float): Temperature parameter for softmax. Default: 0.07
        reduction (str): 'mean' or 'sum'. Default: 'mean'
    """

    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

        if reduction not in ['mean', 'sum']:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")

    def forward(
        self,
        z_i: np.ndarray,
        z_j: np.ndarray
    ) -> float:
        """
        Compute InfoNCE loss for a batch of positive pairs.

        Args:
            z_i (torch.Tensor): Shape [batch_size, embedding_dim]
                Embeddings from first augmentation
            z_j (torch.Tensor): Shape [batch_size, embedding_dim]
                Embeddings from second augmentation (positive pairs with z_i)

        Returns:
            torch.Tensor: Scalar loss value

        Implementation Notes:
        - Assumes z_i and z_j are already normalized to unit sphere
        - Combines i→i+ and i+→i symmetric losses

        Mathematical Steps:
        1. Concatenate z_i and z_j into shape [2*batch_size, embedding_dim]
        2. Compute similarity matrix: logits = z @ z.T / temperature
        3. Create labels: [[1,0,...], [1,0,...]] where position 0 is positive
        4. Apply cross-entropy loss
        """
        raise NotImplementedError(
            "Implement InfoNCE loss:\n"
            "1. Concatenate z_i and z_j along batch dimension\n"
            "2. Compute cosine similarity matrix: logits = (z @ z.T) / self.temperature\n"
            "3. Create labels: i-th anchor's positive is at position (i + N) or (i - N)\n"
            "4. Use F.cross_entropy(logits, labels) for symmetric loss\n"
            "5. Return mean or sum based on self.reduction\n"
            "Hint: The similarity matrix should be [2N, 2N] where N is batch_size"
        )


class ContrastiveDataset:
    """
    Base dataset class for contrastive learning.

    Subclasses should implement __getitem__ to return:
        (x_i, x_j, label)
    where x_i and x_j are two augmented views of the same sample.
    """

    def __init__(self, data, transform=None):
        """
        Args:
            data: Base dataset or list of samples
            transform: Augmentation pipeline that returns two views
        """
        raise NotImplementedError(
            "Subclass must implement __init__ and __getitem__"
        )

    def __getitem__(self, idx):
        """Return two augmented views of sample at idx."""
        raise NotImplementedError(
            "Return tuple: (augmented_sample_1, augmented_sample_2, label)"
        )

    def __len__(self):
        raise NotImplementedError()


class ContrastiveTrainer:
    """
    Base trainer for InfoNCE and similar contrastive objectives.

    Usage:
        trainer = ContrastiveTrainer(model, optimizer, device='cuda')
        for epoch in range(100):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.evaluate(val_loader)
    """

    def __init__(
        self,
        encoder: Module,
        optimizer,
        loss_fn: Module,
        device: str = 'cpu'
    ):
        """
        Args:
            encoder: Feature encoder (e.g., ResNet)
            optimizer: Optimizer (e.g., Adam, SGD)
            loss_fn: Loss function (e.g., InfoNCELoss)
            device: 'cpu' (no GPU support in custom Module system)
        """
        self.encoder = encoder
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss

        Implementation Steps:
        1. Set encoder to training mode
        2. For each batch (x_i, x_j):
           a. Move to device
           b. Encode: z_i = encoder(x_i), z_j = encoder(x_j)
           c. Normalize: z_i /= ||z_i||, z_j /= ||z_j||
           d. Compute loss: L = loss_fn(z_i, z_j)
           e. Backward and optimizer step
        3. Return average loss
        """
        raise NotImplementedError(
            "Implement training loop:\n"
            "1. self.encoder.train()\n"
            "2. Iterate through train_loader\n"
            "3. Encode both views and normalize to unit sphere\n"
            "4. Compute loss and backprop\n"
            "5. Track running average of loss\n"
            "6. Return final average loss"
        )

    def evaluate(self, val_loader) -> float:
        """
        Evaluate on validation set without gradient computation.

        Returns:
            Average validation loss
        """
        raise NotImplementedError(
            "Implement evaluation loop (similar to train_epoch but no backprop)"
        )


# ============================================================================
# Example: How InfoNCE Connects to Other Methods
# ============================================================================

"""
InfoNCE is the foundation for:

1. SimCLR: Extends InfoNCE with:
   - Larger projector head (MLP)
   - Larger batch sizes (4096)
   - Stronger augmentations

2. MoCo: Extends InfoNCE with:
   - Memory bank for more negatives
   - Momentum encoder for consistency

3. BYOL: Conceptually similar but removes negatives:
   - Uses stop_gradient on target network

4. DINO: Uses InfoNCE as template but with:
   - Multiple crops (global + local)
   - Centering and sharpening

Key Insight: All modern self-supervised methods are variants of contrastive
learning that differ in how they handle negatives, batch size, and momentum.
"""
