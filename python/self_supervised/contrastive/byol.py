"""
BYOL: Bootstrap Your Own Latent

A paradigm shift in self-supervised learning: learning representations WITHOUT
explicit negative samples. BYOL learns by having one network predict the output
of another network, with the second network using a stop-gradient operation.

Paper: "Bootstrap your own latent: A new approach to self-supervised Learning"
       https://arxiv.org/abs/2006.07733
       Grill et al. (DeepMind), 2020

Theory:
========
BYOL introduces a fundamentally different approach to self-supervised learning
by removing the explicit contrastive objective (negative samples).

The Key Innovation: Stop Gradient Operation
============================================

Traditional contrastive learning:
  maximize: sim(z_i^+, z_i) - sim(z_i, z_j)
  minimizes: distance to positive pair
             distance from negative pairs

BYOL's approach:
  online_network computes z_online = h_online(x)
  target_network computes z_target = h_target(x')  (with stop_grad on target)
  predictor_network predicts z_target: p = h_predictor(z_online)
  loss = MSE(p, z_target)  where z_target is constant (no gradient flow)

Why No Negative Samples Needed?
===============================

Classical understanding would suggest collapse (all representations identical).
But BYOL avoids this through:

1. **Stop Gradient**: Target network's output is constant during optimization
   - Gradients don't flow back through target network in loss computation
   - Only online network gets gradients
   - Forces online network to learn diverse representations to match target

2. **Momentum Target Network**: Target network updated via EMA
   - Provides a slowly changing target
   - Similar to MoCo but without memory bank
   - θ_target = m·θ_target + (1-m)·θ_online

3. **Asymmetry**: Different augmentations on two views
   - View 1 (online): stronger augmentations
   - View 2 (target): weaker augmentations
   - Creates asymmetry that prevents trivialsolutions

4. **Batch Normalization**: Important stabilizing factor
   - BYOL works even without negatives due to implicit constraints
   - Batch norm statistics reduce instability
   - Without BN, collapse can occur (shown in recent research)

Architecture Comparison:
========================

SimCLR (with negatives):
  [Image] → [Augment 1] → [Encoder] → [Projection] → z_i
            [Augment 2] → [Encoder] → [Projection] → z_j^+

  Loss: InfoNCE(z_i, z_j^+, {z_negatives})

MoCo (with memory bank):
  [Image] → [Augment 1] → [Query Encoder] → [Projection] → z_q
            [Augment 2] → [Momentum Encoder] → [Projection] → z_k

  Loss: Contrastive(z_q, z_k, memory_bank)

BYOL (without negatives - Stop Gradient):
  [Image] → [Augment 1] → [Online Encoder] → [Projection] → [Predictor] → p
            [Augment 2] → [Target Encoder] → [Projection] → z_target (stop_grad)

  Loss: MSE(p, z_target)  where z_target has no gradient

This is a major conceptual shift:
  - Not comparing representations of different samples
  - Not comparing to large set of negatives
  - One network predicts other network's output
  - Symmetry breaking through stop_gradient

BYOL Architecture in Detail:
============================

Online Network:
  f_online(x) ∈ ℝ^2048      [ResNet50 backbone]
         ↓
  g_online(·) ∈ ℝ^128       [Projection: 2-layer MLP with BN]
         ↓
  h_online(·) ∈ ℝ^128       [Predictor: 2-layer MLP without BN (important!)]
         ↓
  p ∈ ℝ^128                  [Final prediction]

Target Network (momentum updated):
  f_target(x') ∈ ℝ^2048     [Same architecture, different weights]
         ↓
  g_target(·) ∈ ℝ^128       [Same architecture as online]
         ↓
  z_target ∈ ℝ^128          [Output (no predictor!)]

  Update rule: θ_target ← m·θ_target + (1-m)·θ_online

Loss and Optimization:
======================

Loss function:
  L = MSE(p, sg(z_target))
    = ||p - sg(z_target)||_2^2 / 2

Where sg(·) is stop-gradient (no gradient flow).

By symmetry, also optimize:
  L_sym = MSE(p', sg(z'_online)) + MSE(p, sg(z_target))

Where:
  - p = h_online(g_online(f_online(x)))
  - z_target = g_target(f_target(x'))  [with sg]
  - p' = h_target(g_target(f_target(x')))
  - z_online = g_online(f_online(x))

Gradient Flow Analysis:
=======================

What happens during backprop?

Forward pass:
  1. Encode x through online: f_online(x)
  2. Project: g_online(f_online(x))
  3. Predict: p = h_online(g_online(f_online(x)))
  4. Encode x' through target: z_target = g_target(f_target(x'))  [no grad]
  5. Compute loss: L = MSE(p, z_target)

Backward pass:
  6. ∂L/∂p = (p - z_target) * ∂p/∂θ_online
  7. Gradients flow: p ← h_online ← g_online ← f_online
  8. No gradients through z_target (stop_grad)
  9. Update θ_online with gradient
  10. Update θ_target ← m·θ_target + (1-m)·θ_online (no gradient update)

Key insight: Target network gets no gradient flow, only momentum update!

Why Doesn't It Collapse?
=======================

This is the critical question. Why don't all online representations become identical?

Research findings (BYOL with PyTorch 1.9+):
1. Batch Normalization prevents collapse
   - BN statistics vary across batch → implicit diversity
   - Different batches see different statistics → prevents uniform collapse
   - Without BN: representation collapse occurs

2. Stop Gradient Mechanism
   - Forces online to learn diverse features
   - Can't cheat by making all representations same
   - Momentum update ensures target is different

3. Momentum Encoder Consistency
   - Target provides slowly changing target
   - Online must find diverse way to match it
   - If all same: MSE loss would increase

4. Asymmetric Augmentations
   - Different views processed differently
   - Creates natural diversity enforcement

Important caveat:
  Some research suggests batch normalization is more critical than initially thought.
  BYOL works across GPUs but needs careful BN setup.

Advantages of BYOL:
===================
  + No need for large batch sizes
  + No memory bank required
  + Simple and elegant formulation
  + Works with small batch sizes (32-64)
  + Momentum mechanism alone is sufficient
  + No explicit negative sampling
  + Efficient use of compute

Disadvantages of BYOL:
======================
  + Requires careful batch norm setup
  - Collapse can occur without proper implementation
  - More sensitive to hyperparameters
  - Momentum encoder adds slight complexity (less than MoCo)
  - Still requires two networks

Practical Implementation:
=========================

Key implementation details:

1. **Stop Gradient**: Must detach target network output
   z_target = model_target(x).detach()

2. **Momentum Update**: After backward pass
   update_target_network(theta_online, theta_target, m=0.999)

3. **Predictor Head**: Special considerations
   - No batch norm after output of predictor
   - Hidden layer HAS batch norm
   - Usually: output_size = output_size (no projection)

4. **Batch Normalization Setup**:
   - Online network: full BN
   - Target network: full BN (separate stats)
   - Both networks: synchronized BN if multi-GPU
   - This is subtle but important

5. **Augmentations**:
   - Not as critical as SimCLR
   - Still benefit from strong augmentations
   - Even weak augmentations work (unlike SimCLR)

Hyperparameters:
================

  momentum (m): 0.999 (controls target network update speed)
  learning_rate: 0.2 (adjusted for batch size)
  weight_decay: 1e-6
  batch_size: 256 (can be smaller than SimCLR)
  predictor_hidden_dim: 4096
  projection_dim: 256

Unlike SimCLR, BYOL is relatively robust to these choices.

BYOL vs SimCLR vs MoCo:
=======================

                    BYOL        SimCLR      MoCo
Paradigm:          Asymmetric  Symmetric   Asymmetric
Negatives:         None        In-batch    Memory bank
#Negatives:        0           2N-2        K (~65k)
Batch size needs:  Small (64)  Large (256) Medium (256)
Momentum encoder:  Yes         No          Yes
Memory bank:       No          No          Yes
Collapse risk:     Yes*        No          No

*BYOL collapse prevented by BN, not by loss function

SimCLR Strengths:
  - Conceptually simple (maximize sim to positive, minimize to negatives)
  - Well understood why it works
  - Negatives provide clear learning signal

MoCo Strengths:
  - Efficient single-GPU training
  - Hard negatives from history
  - Clear information-theoretic grounding

BYOL Strengths:
  - Simplest loss function (just MSE)
  - Works with small batch sizes
  - No memory bank needed
  - Momentum mechanism elegant

Recent Developments:
====================

SimSiam (2020): Simplified BYOL
  - Removes momentum network entirely
  - Only uses stop_gradient on projections
  - MSE + Stop gradient sufficient
  - Even simpler than BYOL

VICReg (2022): Variance-Covariance Regularization
  - Uses variance, covariance, and correlation regularization
  - Replaces momentum and stop_gradient with regularization
  - Different approach but related philosophy

Understanding BYOL Better:
==========================

The conceptual leap in BYOL:
1. Traditional ML: Learn from labeled data (supervised)
2. Self-supervised (contrastive): Learn from relationships between samples
3. BYOL: Learn by predicting what another network predicts

This represents a different paradigm:
  - Not comparison-based (negatives)
  - Not information maximization (InfoNCE)
  - But prediction-based with asymmetry breaking

The role of batch norm cannot be overstated:
  - Empirically found critical for BYOL
  - Explains why it avoids collapse
  - Recent research continues investigating this

Mathematical Interpretation:
  BYOL can be viewed as implicit EM algorithm
  where batch norm enforces implicit constraint.
"""

import numpy as np
from typing import Tuple, Optional
from copy import deepcopy
from python.nn_core import Module, Parameter


class BYOLProjectionHead(Module):
    """
    Projection head for BYOL online network.

    Architecture: 2-layer MLP with batch normalization
    between layers but NOT after output.

    Important: Different from SimCLR's projection head
    which has no BN after output. BYOL has the same structure.

    Formula:
      z = BN(Linear(ReLU(BN(Linear(h)))))

    Args:
        input_dim: Input feature dimension (e.g., 2048)
        hidden_dim: Hidden dimension (typically same as input_dim)
        output_dim: Output projection dimension (e.g., 256)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 256
    ):
        """
        Args:
            input_dim: Dimension of encoder output
            hidden_dim: Hidden layer dimension
            output_dim: Output projection dimension
        """
        super().__init__()
        raise NotImplementedError(
            "Implement BYOL projection head:\n"
            "1. First linear: input_dim → hidden_dim\n"
            "2. BatchNorm1d\n"
            "3. ReLU\n"
            "4. Second linear: hidden_dim → output_dim\n"
            "5. NO BatchNorm after final output (important!)\n"
            "Formula: output_dim = BN(Linear(ReLU(BN(Linear(input_dim)))))"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to embedding space."""
        raise NotImplementedError()


class BYOLPredictor(Module):
    """
    Predictor head for BYOL online network.

    The online network additionally has a predictor head h that predicts
    the target network's projections.

    Architecture: 2-layer MLP with batch normalization
    but NO activation after output.

    Formula:
      p = Linear(ReLU(BN(Linear(z))))

    This is applied to the projection output to get final prediction.

    Args:
        input_dim: Input dimension (projection dimension, e.g., 256)
        hidden_dim: Hidden dimension (typically 2x input_dim, e.g., 4096)
        output_dim: Output dimension (typically same as input_dim, e.g., 256)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 4096,
        output_dim: int = 256
    ):
        """
        Args:
            input_dim: Dimension of input (projection output)
            hidden_dim: Hidden layer dimension
            output_dim: Output prediction dimension
        """
        super().__init__()
        raise NotImplementedError(
            "Implement BYOL predictor head:\n"
            "1. First linear: input_dim → hidden_dim\n"
            "2. BatchNorm1d\n"
            "3. ReLU\n"
            "4. Second linear: hidden_dim → output_dim\n"
            "5. NO BatchNorm or activation after final output\n"
            "Formula: p = Linear(ReLU(BN(Linear(z))))"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict target network output."""
        raise NotImplementedError()


class BYOLOnlineNetwork(Module):
    """
    Online network used for computing predictions.

    Architecture:
      Input → [Encoder f(·)] → h ∈ ℝ^2048
             → [Projection g(·)] → z ∈ ℝ^256
             → [Predictor h(·)] → p ∈ ℝ^256

    The online network is the only network that gets gradient updates.
    The target network is updated via exponential moving average.

    Usage:
      online_net = BYOLOnlineNetwork()
      features = online_net.get_features(x)  # Get h
      projections = online_net.get_projections(x)  # Get z
      predictions = online_net(x)  # Get p (full forward)
    """

    def __init__(
        self,
        encoder: Module,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        predictor_hidden_dim: int = 4096
    ):
        """
        Args:
            encoder: Base encoder network (e.g., ResNet50)
            projection_dim: Dimension of projection output
            hidden_dim: Hidden dimension of projection head
            predictor_hidden_dim: Hidden dimension of predictor head
        """
        super().__init__()
        raise NotImplementedError(
            "Implement online network:\n"
            "1. Store encoder\n"
            "2. Create projection head with appropriate dimensions\n"
            "3. Create predictor head\n"
            "4. Ensure output_dim of projection matches input_dim of predictor"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions p for input x."""
        raise NotImplementedError()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder features h (before projection)."""
        raise NotImplementedError()

    def get_projections(self, x: torch.Tensor) -> torch.Tensor:
        """Get projections z (before predictor)."""
        raise NotImplementedError()


class BYOLTargetNetwork(Module):
    """
    Target network that provides targets for online network prediction.

    The target network has the same architecture as the online network
    but WITHOUT the predictor head.

    Architecture:
      Input → [Encoder f(·)] → h ∈ ℝ^2048
             → [Projection g(·)] → z ∈ ℝ^256
             (No predictor head)

    Update mechanism: Exponential Moving Average (EMA)
      θ_target ← m·θ_target + (1-m)·θ_online

    Where m ≈ 0.999 means target network updates slowly.

    Important: Outputs of target network are detached (no gradients).
    """

    def __init__(
        self,
        online_network: BYOLOnlineNetwork,
        momentum: float = 0.999
    ):
        """
        Args:
            online_network: Online network to use as template
            momentum: Momentum coefficient for EMA update
        """
        super().__init__()
        raise NotImplementedError(
            "Implement target network:\n"
            "1. Copy architecture of online_network\n"
            "2. Remove predictor head (target only has encoder + projection)\n"
            "3. Initialize with same weights as online_network\n"
            "4. Store momentum coefficient"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get target projections (detached from gradient).

        Args:
            x: Input tensor

        Returns:
            Projections z (with no_grad context)
        """
        raise NotImplementedError(
            "Implement forward pass:\n"
            "1. Pass x through encoder\n"
            "2. Pass through projection head\n"
            "3. Return z (no gradient tracking needed)"
        )

    def update_weights(self, online_network: BYOLOnlineNetwork):
        """
        Update target network weights via EMA.

        Implements: θ_target ← m·θ_target + (1-m)·θ_online

        Args:
            online_network: Online network to get weights from
        """
        raise NotImplementedError(
            "Implement EMA update:\n"
            "For each (param_target, param_online) pair:\n"
            "  param_target.data = m * param_target.data + (1-m) * param_online.data\n"
            "Use @torch.no_grad() decorator"
        )


class BYOLModel(Module):
    """
    Complete BYOL model with online and target networks.

    The model maintains two networks:
    1. Online network: Updated via gradient
    2. Target network: Updated via EMA

    Usage:
      model = BYOLModel(encoder)
      p = model.predict(x_online)  # Get prediction from online network
      with torch.no_grad():
          z_target = model.get_target(x_target)  # Get target projection
      loss = F.mse_loss(p, z_target)
      loss.backward()
      model.update_target()  # Update target network via EMA
    """

    def __init__(
        self,
        encoder: Module,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        predictor_hidden_dim: int = 4096,
        momentum: float = 0.999
    ):
        """
        Args:
            encoder: Base encoder network
            projection_dim: Projection output dimension
            hidden_dim: Projection head hidden dimension
            predictor_hidden_dim: Predictor head hidden dimension
            momentum: EMA momentum for target network
        """
        super().__init__()
        raise NotImplementedError(
            "Implement BYOL model:\n"
            "1. Create online network with all heads\n"
            "2. Create target network (copy of online without predictor)\n"
            "3. Store momentum coefficient"
        )

    def forward(
        self,
        x_online: np.ndarray,
        x_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for both networks.

        Args:
            x_online: Input for online network
            x_target: Input for target network

        Returns:
            p: Predictions from online network
            z_target: Projections from target network (detached)
        """
        raise NotImplementedError(
            "Implement forward pass:\n"
            "1. p = online_network(x_online)\n"
            "2. z_target = target_network(x_target)  (with no_grad)\n"
            "3. Return p, z_target"
        )

    def update_target_network(self):
        """Update target network weights via EMA."""
        raise NotImplementedError(
            "Call target_network.update_weights(online_network)"
        )


class BYOLLoss(Module):
    """
    BYOL loss function.

    Simple MSE between predictions and (detached) target projections.

    L = 2 - 2 * <p, z_target> / (||p|| ||z_target||)

    This is equivalent to:
    L = MSE(p / ||p||, z_target / ||z_target||)

    But numerically more stable using cosine similarity.

    Args:
        reduction: 'mean' or 'sum'
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: How to reduce loss across batch
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, p: np.ndarray, z: np.ndarray) -> float:
        """
        Compute BYOL loss.

        Args:
            p: Predictions from online network [batch_size, dim]
            z: Target projections (detached) [batch_size, dim]

        Returns:
            Scalar loss value

        Implementation:
        1. Normalize both p and z to unit sphere
        2. Compute cosine similarity: sim = p @ z.T (after normalization)
        3. Loss = 2 - 2*sim (simplified form)
        4. Or equivalently: MSE(normalize(p), normalize(z))

        Note: Can use either formulation - both equivalent mathematically
        """
        raise NotImplementedError(
            "Implement BYOL loss:\n"
            "Option 1 (Cosine similarity based):\n"
            "  1. Normalize: p_n = F.normalize(p, dim=1)\n"
            "  2. Normalize: z_n = F.normalize(z, dim=1)\n"
            "  3. Compute dot product: similarity = (p_n * z_n).sum(dim=1)\n"
            "  4. Loss = 2 - 2*similarity  (or -(2*similarity).mean())\n"
            "\n"
            "Option 2 (MSE based, equivalent):\n"
            "  1. Normalize both tensors\n"
            "  2. Loss = F.mse_loss(p_n, z_n)\n"
            "\n"
            "Hint: Option 1 is more numerically stable"
        )


class BYOLTrainer:
    """
    Trainer for BYOL self-supervised learning.

    Handles:
    - Training loop with gradient updates
    - Target network EMA updates
    - Symmetric loss (both directions)
    - Checkpoint saving/loading

    Key difference from SimCLR/MoCo:
    - No negative samples needed
    - Simpler loss function
    - Can work with smaller batch sizes
    - Critical: Batch normalization is essential

    Usage:
        model = BYOLModel(encoder)
        trainer = BYOLTrainer(model, train_loader, device='cuda')
        for epoch in range(300):
            train_loss = trainer.train_epoch()
    """

    def __init__(
        self,
        model: BYOLModel,
        optimizer,
        train_loader,
        loss_fn: BYOLLoss,
        device: str = 'cpu'
    ):
        """
        Args:
            model: BYOLModel instance
            optimizer: Optimizer (Adam or SGD)
            train_loader: Training data loader
            loss_fn: BYOLLoss instance
            device: 'cpu' (no GPU support in custom Module system)
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss

        Training Loop:
        1. Set model to training mode
        2. For each batch (x_i, x_j):
           a. Forward pass: p_i, z_j = model(x_i, x_j)
           b. Compute loss: L_i = loss(p_i, z_j)
           c. Forward pass (symmetric): p_j, z_i = model(x_j, x_i)
           d. Compute loss: L_j = loss(p_j, z_i)
           e. Total loss: L = (L_i + L_j) / 2
           f. Backward and optimizer step
           g. Update target network weights via EMA
        3. Return average loss

        Important Notes:
        - Symmetric loss: optimize both directions
        - Update target network AFTER backward pass
        - EMA update is gradual (no gradient flow)
        - Batch normalization must be enabled for stability

        Why symmetric loss?
        - p_i predicts z_j (online on x_i predicts target on x_j)
        - p_j predicts z_i (online on x_j predicts target on x_i)
        - Both contribute equally to the objective
        """
        raise NotImplementedError(
            "Implement BYOL training loop:\n"
            "1. self.model.train()\n"
            "2. For each batch (x_i, x_j):\n"
            "   a. Forward: p_i, z_j = model(x_i, x_j)\n"
            "   b. Loss_1: loss_fn(p_i, z_j)\n"
            "   c. Forward: p_j, z_i = model(x_j, x_i)\n"
            "   d. Loss_2: loss_fn(p_j, z_i)\n"
            "   e. Total: loss = (loss_1 + loss_2) / 2\n"
            "   f. Backward and optimizer step\n"
            "   g. Update target: model.update_target_network()\n"
            "3. Return average loss\n"
            "\n"
            "Hint: Symmetric loss ensures both views are utilized equally"
        )

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint."""
        raise NotImplementedError()

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        raise NotImplementedError()


# ============================================================================
# Understanding BYOL: Key Concepts
# ============================================================================

"""
Why BYOL Represents a Paradigm Shift:

1. **Different Objective**:
   - SimCLR/MoCo: Maximize contrast (similarity to positive, dissimilarity to negatives)
   - BYOL: Predict target network output (asymmetric, no negatives)

2. **Theoretical Implications**:
   - Traditional theory: Need negatives to prevent collapse
   - BYOL: Collapse prevented by architecture (asymmetry, stop_gradient, BN)
   - Shows that contrastive learning may not be necessary

3. **Practical Benefits**:
   - Works with small batch sizes (no large N needed)
   - No memory bank (simpler implementation)
   - Elegant loss function (just MSE)
   - Momentum mechanism sufficient for consistency

4. **Research Insights**:
   - Batch normalization more critical than originally thought
   - Stop gradient creates implicit constraint
   - Predictor head crucial (prevents trivial solutions)
   - Shows power of asymmetric objectives

Critical Implementation Details:

1. **Stop Gradient**: Must be applied to target network output
   z_target = model_target(x).detach()

2. **Batch Normalization**: Cannot be disabled
   - Batch norm statistics provide implicit constraint
   - Prevents representation collapse
   - Works across batches for regularization

3. **Predictor Head**: Not just projection
   - Adds non-linearity between projection and loss
   - Allows online network to learn complex features
   - Without predictor: collapse or poor performance

4. **Momentum Update**: Gradual change is key
   - m = 0.999: target updates at rate ~1000 batches
   - Too fast (m < 0.99): instability
   - Too slow (m > 0.9999): sluggish adaptation

5. **Symmetric Loss**: Both directions matter
   - Not strictly necessary but improves performance
   - Ensures utilization of both augmented views
   - Standard practice in BYOL implementation

Debugging Common Issues:

1. Representation Collapse:
   - Cause: Batch norm disabled or batch size too small
   - Solution: Ensure batch norm enabled, batch size >= 32

2. Training Instability:
   - Cause: Momentum too high or stop_grad incorrectly applied
   - Solution: Use m=0.999, verify no gradients flow through target

3. Poor Performance:
   - Cause: Predictor head missing or incorrect architecture
   - Solution: Ensure predictor exists and has sufficient capacity

4. Memory Issues:
   - Cause: Two full networks in memory
   - Solution: Can reduce architecture size or use gradient checkpointing
"""
