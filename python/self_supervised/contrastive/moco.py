"""
Momentum Contrast (MoCo): Unsupervised Representation Learning via Hard Negative Mining

MoCo introduces a momentum encoder and memory bank to maintain a large, consistent
set of negative samples without requiring enormous batch sizes. This enables
efficient self-supervised learning on single GPUs while preserving gradient quality.

Paper: "Momentum Contrast for Unsupervised Visual Representation Learning"
       https://arxiv.org/abs/1911.05722
       He et al. (Facebook AI Research), 2019

Theory:
========
The key innovation of MoCo is addressing the "queue" problem in contrastive learning:

Problem:
  - SimCLR needs large batch sizes (4096) for sufficient negatives
  - Memory intensive, requires many GPUs
  - What if we could decouple batch size from number of negatives?

Solution: Memory Bank + Momentum Encoder
  1. Maintain a queue Q of K old representations (K=65536 or larger)
  2. Use momentum encoder to update queue (slower updates → consistency)
  3. Query encoder learns from current batch + K negatives from queue
  4. Momentum encoder: θ_m = m·θ_m + (1-m)·θ_q, where m ≈ 0.999

Architecture Overview:
=======================

Query Network θ_q:        Updated every iteration via gradient
                          (from current batch)
                          ↓
                   [Encoder f(·)]
                   [Projection g(·)]
                          ↓
                   [Feature Queue Q]

Momentum Network θ_m:     Updated slowly via momentum
                          (EMA with momentum m ≈ 0.999)
                          ↓
                   [Encoder f(·)]
                   [Projection g(·)]
                          ↓
                   [Memory Bank Q] ← receives new negatives

Training Loop:
1. Sample mini-batch of N images
2. Encode with query encoder: z_q = encoder_q(x_q)
3. Encode with momentum encoder: z_k = encoder_m(x_k) where x_k ~ X
4. Update momentum encoder: θ_m ← m·θ_m + (1-m)·θ_q
5. Retrieve queue Q of K old representations
6. Compute contrastive loss: L = CrossEntropy(logits, labels)
   where logits = [sim(z_q, z_k), sim(z_q, Q)] / τ
7. Enqueue z_k → push oldest queue entries out
8. Dequeue oldest entries

Key Insight - Momentum Consistency:
===================================

Why use momentum encoder instead of just old encoder weights?

1. **Consistency**: Momentum encoder changes slowly
   - Encoder θ_q changes via gradient each iteration
   - Encoder θ_m changes via exponential moving average
   - Slow change → representations in queue remain consistent
   - If θ_m updated too fast → queue becomes incoherent

2. **Gradient Flow**:
   - Query encoder gets gradients from current batch + queue
   - Momentum encoder only gets gradients implicitly
   - Momentum update formula: θ_m = m·θ_m + (1-m)·θ_q
   - Conceptually: θ_m is dragged slowly toward θ_q

3. **Memory Efficiency**:
   - Don't need to store gradients for momentum encoder
   - Can use large K (65536) without explosion in memory
   - K >> batch_size is the key advantage

Momentum Parameter m:
=====================

Effect of momentum value m ∈ [0, 1]:

  m → 1 (e.g., 0.999):
    + Queue remains very consistent
    + Stronger negatives (old representations)
    - May be too stale (representation mismatch)
    - Slower adaptation to query encoder changes

  m → 0 (e.g., 0.9):
    + Queue updates faster, fresher negatives
    - Less consistent (mismatch between query and momentum)
    - Harder to train (larger representation shift)

  Typical: m = 0.999 (from paper)
  - Per iteration: θ_m := 0.999·θ_m + 0.001·θ_q
  - After N iterations: queue has mix of representations
  - Queue mixes representations from ~1000 iterations ago

Queue Size K:
==============

Typical values: K = 65536 (2^16)

Effect of K:
  - K = 256: Barely better than batch size
  - K = 16384: Good performance
  - K = 65536: Near-optimal (saturates)
  - K = 131072: Marginal improvement over 65536

Trade-off:
  - Larger K → more negatives → stronger gradients
  - Larger K → more memory required (K × embedding_dim floats)
  - K = 65536 good balance for typical GPUs

Queue FIFO Mechanism:
======================

The queue operates as a first-in-first-out (FIFO) buffer:

Queue State: [old_repr_1, old_repr_2, ..., old_repr_K]

With new batch providing z_k:
1. Concatenate: [z_k, old_repr_2, ..., old_repr_K]
2. Keep latest K: [z_k, old_repr_2, ..., old_repr_{K}]  [remove oldest]
3. After N batches: queue contains representations from ~N/batch_size iterations ago

Important: Queue provides hard negatives (from same samples but old encoder)

MoCo vs SimCLR Comparison:
===========================

                     MoCo              SimCLR
Memory (K=65k):      65k × 128 floats  4096 × 128 floats
Batch size:          256               4096
Negatives per iter:  65536             4096
Training time:       1 GPU * 200 eps   8 GPUs * 100 eps
ImageNet Acc:        60.6% (backbone)  69.3%

MoCo advantages:
  + Works on single GPU with modest batch size
  + Queue provides hard negatives from history
  + More memory efficient overall
  + Simpler to implement

SimCLR advantages:
  + All negatives from current batch (no staleness)
  + No momentum encoder complexity
  + Slightly better performance
  + Easier to understand conceptually

Both are highly effective self-supervised methods.

Implementation Details:
=======================

1. **Queue as Circular Buffer**:
   - Implement using numpy operations
   - Or maintain explicit queue tensor
   - Track pointer to determine where to enqueue

2. **Momentum Update**:
   - After each iteration: copy query weights to momentum with EMA
   - Option 1: m * m_param + (1-m) * q_param for each param
   - Option 2: Use @torch.no_grad() for momentum update
   - No gradients flow through momentum encoder

3. **Consistency in Batch Normalization**:
   - Momentum encoder uses its own batch norm
   - Different statistics per encoder (not shared)
   - Important for contrastive learning to work

4. **Shuffling and Distributed Training**:
   - Shuffle samples before momentum encoding (prevents trivial solutions)
   - For multi-GPU: shuffle batch so momentum encoder sees different samples
   - Important: Momentum encoder must not see query samples

Advanced Topics:
================

1. **Negative Mining**:
   - Unlike SimCLR where all batch items are negatives
   - MoCo negatives from queue are harder (from different time)
   - Harder negatives provide stronger learning signal

2. **Memory Bank Consistency**:
   - Why not use query encoder to encode all queue entries?
   - Too expensive (would require re-encoding entire queue)
   - Momentum encoder provides smooth approximation

3. **Shuffling Strategy** (Important!):
   - Before momentum encoding: shuffle batch order
   - This prevents exploitation of batch ordering
   - After momentum encoding: restore original order
   - Prevents the model from learning trivial solutions based on order

4. **Large Momentum Networks**:
   - Full encoder + momentum encoder = 2× parameters
   - Can be mitigated by using momentum encoder less frequently
   - Or checkpoint-based approaches

Related Work and Variants:
==========================

MoCo-v2 (2020):
  - Larger hidden dimension in projection head
  - Cosine learning rate scheduler
  - Longer training (800 epochs vs 200)
  - Better downstream performance

MoCo-v3 (2021):
  - Removes memory queue (uses large batch size again)
  - Uses BYOL-style momentum without explicit negatives
  - Momentum contrast + momentum predictor
  - Combines best of MoCo and BYOL

SwAV (Swapped Assignment with Views):
  - Alternative to momentum: uses clustering
  - Replaces memory bank with cluster assignments
  - Different approach but similar efficiency

Key Equation (MoCo Loss):
=========================

L = -log[exp(sim(q, k_+) / τ) / (exp(sim(q, k_+) / τ) + Σ_i exp(sim(q, k_i) / τ))]

Where:
  - q: query representation (from query encoder)
  - k_+: positive key (from momentum encoder on same sample)
  - k_i: negative keys from queue (old representations)
  - sim(q, k) = q^T k / (||q|| ||k||)  [cosine similarity]
  - τ: temperature parameter

Intuition: "How well can I distinguish the positive from all queue negatives?"
"""

import numpy as np
from typing import Tuple, Optional, Callable
from collections import deque
from python.nn_core import Module, Parameter


class MemoryBank(Module):
    """
    Efficient implementation of a queue-based memory bank for MoCo.

    Maintains a FIFO queue of K negative representations.
    Updates queue with new representations and removes oldest ones.

    Usage:
        memory = MemoryBank(dim=128, K=65536)
        # First iteration
        memory.enqueue(z_k)  # Add new representations
        negatives = memory.queue  # Get all negatives
    """

    def __init__(self, dim: int, K: int = 65536):
        """
        Args:
            dim: Dimension of stored representations
            K: Size of queue (typical: 65536)
        """
        super().__init__()
        self.K = K
        self.dim = dim
        self.ptr = 0  # Pointer to next insertion position

        # Initialize queue as numpy array
        self.queue = np.random.randn(K, dim).astype(np.float32)
        # Normalize queue
        self.queue = self.queue / (np.linalg.norm(self.queue, axis=1, keepdims=True) + 1e-8)

    def enqueue(self, batch: torch.Tensor):
        """
        Add batch to queue and remove oldest entries.

        Args:
            batch: New representations [batch_size, dim]

        Implementation:
        1. Normalize batch
        2. Determine insertion positions
        3. Handle wrap-around if necessary
        4. Update pointer
        """
        raise NotImplementedError(
            "Implement FIFO queue enqueue:\n"
            "1. Normalize input batch: batch = F.normalize(batch, dim=1)\n"
            "2. Get insertion range: ptr to ptr+batch_size (handle wrap-around)\n"
            "3. Update self.queue at those positions\n"
            "4. Update self.ptr: ptr = (ptr + batch_size) % K\n"
            "Hint: Handle wrap-around using modulo when inserting"
        )

    def get_queue(self) -> np.ndarray:
        """Return current queue [K, dim]."""
        return self.queue.copy()


class MomentumEncoder(Module):
    """
    Momentum encoder that tracks query encoder via exponential moving average.

    The momentum encoder is updated slowly (m ≈ 0.999) to maintain consistency
    in the representation space while allowing gradual adaptation.

    Usage:
        query_encoder = ResNet50()
        momentum_encoder = MomentumEncoder(query_encoder, m=0.999)

        # Training loop
        z_q = query_encoder(x)
        with torch.no_grad():
            z_m = momentum_encoder.encoder(x)
        momentum_encoder.update_params()  # Update momentum weights
    """

    def __init__(self, query_encoder: Module, momentum: float = 0.999):
        """
        Args:
            query_encoder: Query encoder network
            momentum: Momentum coefficient (typically 0.999)
        """
        super().__init__()
        self.momentum = momentum

        # Copy query encoder architecture
        self.encoder = self._create_momentum_encoder(query_encoder)

        # Initialize momentum encoder with query encoder weights
        self._copy_weights(query_encoder, self.encoder)

    def _create_momentum_encoder(self, query_encoder: Module) -> Module:
        """Create momentum encoder with same architecture as query encoder."""
        raise NotImplementedError(
            "Create a copy of query_encoder:\n"
            "1. Use nn.Sequential or similar to create momentum encoder\n"
            "2. Initialize with same architecture as query_encoder\n"
            "3. Weights will be copied separately"
        )

    def _copy_weights(self, source: Module, target: Module):
        """Copy weights from source to target encoder."""
        raise NotImplementedError(
            "Copy all parameters from source to target:\n"
            "for p_src, p_tgt in zip(source.parameters(), target.parameters()):\n"
            "    p_tgt.data.copy_(p_src.data)"
        )

    def update_params(self, query_encoder: Module):
        """
        Update momentum encoder parameters via EMA.

        Implements: θ_m ← m·θ_m + (1-m)·θ_q

        Args:
            query_encoder: Query encoder to take parameters from
        """
        raise NotImplementedError(
            "Update momentum encoder weights via EMA:\n"
            "For each parameter pair (query, momentum):\n"
            "  momentum.data = m * momentum.data + (1-m) * query.data\n"
            "Use @torch.no_grad() to prevent gradient computation"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through momentum encoder (no gradients).

        Args:
            x: Input tensor

        Returns:
            Encoded representation
        """
        return self.encoder(x)


class ShuffleAndUnshuffle:
    """
    Utility for shuffling batch before momentum encoding.

    Important for preventing trivial solutions where model learns
    from batch ordering rather than image content.

    Usage:
        shuffler = ShuffleAndUnshuffle()
        x_shuffled, shuffle_idx = shuffler.shuffle(x)
        z_m = momentum_encoder(x_shuffled)
        z_m = shuffler.unshuffle(z_m, shuffle_idx)
    """

    def __init__(self, world_size: int = 1, rank: int = 0):
        """
        Args:
            world_size: Number of GPUs (for distributed training)
            rank: Current GPU rank
        """
        self.world_size = world_size
        self.rank = rank

    def shuffle(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shuffle batch samples (important for multi-GPU).

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Shuffled tensor, indices for unshuffling
        """
        raise NotImplementedError(
            "Implement batch shuffling:\n"
            "1. Get batch size N\n"
            "2. Create random permutation indices\n"
            "3. Return x[perm_idx], perm_idx for later unshuffling"
        )

    def unshuffle(self, x: np.ndarray, shuffle_idx: np.ndarray) -> np.ndarray:
        """
        Unshuffle to restore original order.

        Args:
            x: Shuffled tensor
            shuffle_idx: Indices from shuffle operation

        Returns:
            Unshuffled tensor in original order
        """
        raise NotImplementedError(
            "Implement unshuffling:\n"
            "1. Create inverse permutation: inv_idx[shuffle_idx] = range(N)\n"
            "2. Return x[inv_idx]"
        )


class MoCoModel(Module):
    """
    Complete MoCo model with query encoder, momentum encoder, and projection heads.

    Architecture:
        Query Network:
          Input → [Encoder] → [Projection] → z_q

        Momentum Network:
          Input → [Momentum Encoder] → [Momentum Projection] → z_k

        Memory Bank:
          Stores K old representations from momentum encoder
    """

    def __init__(
        self,
        encoder: Module,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        queue_size: int = 65536,
        momentum: float = 0.999
    ):
        """
        Args:
            encoder: Base encoder (e.g., ResNet50 backbone)
            projection_dim: Output dimension of projection head
            hidden_dim: Hidden dimension of projection head
            queue_size: Size of memory bank queue
            momentum: Momentum coefficient for updating momentum encoder
        """
        super().__init__()
        raise NotImplementedError(
            "Implement MoCo model:\n"
            "1. Create query encoder with projection head\n"
            "2. Create momentum encoder (copy of query encoder)\n"
            "3. Initialize momentum encoder with same weights\n"
            "4. Create memory bank\n"
            "5. Store momentum and queue_size as attributes"
        )

    def forward(
        self,
        x_q: np.ndarray,
        x_k: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for both query and momentum encoders.

        Args:
            x_q: Query images [batch_size, 3, H, W]
            x_k: Momentum images [batch_size, 3, H, W]

        Returns:
            z_q: Query projections [batch_size, projection_dim]
            z_k: Momentum projections [batch_size, projection_dim]
        """
        raise NotImplementedError(
            "Implement forward pass:\n"
            "1. Encode x_q with query encoder and projection head\n"
            "2. Normalize z_q\n"
            "3. Encode x_k with momentum encoder (no_grad)\n"
            "4. Normalize z_k\n"
            "5. Return z_q, z_k"
        )

    def update_momentum_encoder(self):
        """Update momentum encoder parameters via EMA."""
        raise NotImplementedError(
            "Update momentum encoder weights:\n"
            "Call self.momentum_encoder.update_params(self.query_encoder)"
        )


class MoCoLoss(Module):
    """
    Contrastive loss for MoCo.

    Similar to SimCLR but uses memory bank negatives instead of batch negatives.

    L = -log[exp(sim(q, k_+) / τ) / (exp(sim(q, k_+) / τ) + Σ_i exp(sim(q, k_i) / τ))]

    Where:
      - q: query representation
      - k_+: positive key (from momentum encoder, same sample as query)
      - k_i: all keys in memory bank queue
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter (typical: 0.07)
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_q: np.ndarray,
        z_k: np.ndarray,
        queue: np.ndarray
    ) -> float:
        """
        Compute MoCo loss.

        Args:
            z_q: Query projections [batch_size, dim]
            z_k: Key projections [batch_size, dim] (positive pair with z_q)
            queue: Memory bank queue [queue_size, dim]

        Returns:
            Scalar loss value

        Implementation:
        1. Compute similarity between query and positive key
        2. Compute similarities between query and all queue elements
        3. Concatenate: logits = [sim(q, k_+), sim(q, queue)] / τ
        4. Create labels: first position (k_+) is positive, rest are negatives
        5. Apply cross-entropy loss
        """
        raise NotImplementedError(
            "Implement MoCo loss:\n"
            "1. Compute logits with positive: pos_logits = (z_q @ z_k.T) / τ [batch_size, 1]\n"
            "2. Compute logits with queue: neg_logits = (z_q @ queue.T) / τ [batch_size, K]\n"
            "3. Concatenate: logits = torch.cat([pos_logits, neg_logits], dim=1)\n"
            "4. Labels: all zeros (first column is positive)\n"
            "5. Return cross_entropy loss"
        )


class MoCoTrainer:
    """
    Trainer for MoCo self-supervised learning.

    Handles:
    - Loading data with augmentation
    - Training loop with momentum updates
    - Memory bank management
    - Batch shuffling (prevents trivial solutions)

    Usage:
        model = MoCoModel(encoder)
        trainer = MoCoTrainer(model, train_loader, device='cuda')
        for epoch in range(200):
            train_loss = trainer.train_epoch()
    """

    def __init__(
        self,
        model: MoCoModel,
        optimizer,
        train_loader,
        loss_fn: MoCoLoss,
        device: str = 'cpu',
        world_size: int = 1,
        rank: int = 0
    ):
        """
        Args:
            model: MoCoModel instance
            optimizer: Optimizer (SGD with momentum)
            train_loader: Training data loader
            loss_fn: MoCoLoss instance
            device: 'cpu' (no GPU support in custom Module system)
            world_size: Number of GPUs (for distributed training)
            rank: Current GPU rank
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.shuffler = ShuffleAndUnshuffle(world_size, rank)

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss

        Training Loop:
        1. Set model to training mode
        2. For each batch (x_q, x_k):
           a. Shuffle x_k to prevent trivial solutions
           b. Forward pass: z_q, z_k = model(x_q, x_k)
           c. Get current queue from memory bank
           d. Compute loss using z_q, z_k, and queue
           e. Backward and optimizer step
           f. Update momentum encoder weights
           g. Enqueue z_k to memory bank (dequeue oldest)
        3. Return average loss

        Important Notes:
        - Must shuffle x_k before momentum encoding (different from x_q)
        - Update momentum encoder AFTER backward (not before)
        - Handle memory bank updates carefully
        """
        raise NotImplementedError(
            "Implement MoCo training loop:\n"
            "1. self.model.train()\n"
            "2. For each batch (x_q, x_k):\n"
            "   a. Shuffle x_k and keep track of shuffle indices\n"
            "   b. Forward: z_q, z_k = model(x_q, x_k)\n"
            "   c. Unshuffle z_k back to original order\n"
            "   d. Get queue from memory bank\n"
            "   e. Compute loss: loss_fn(z_q, z_k, queue)\n"
            "   f. Backward and optimizer step\n"
            "   g. Update momentum encoder\n"
            "   h. Enqueue z_k to memory bank\n"
            "   i. Track running loss\n"
            "3. Return average loss"
        )


# ============================================================================
# Key Insights about MoCo
# ============================================================================

"""
Why MoCo is Efficient:

1. **Decouples Batch Size from #Negatives**:
   - SimCLR: #negatives = batch_size (requires large batches)
   - MoCo: #negatives = queue_size >> batch_size
   - Can use batch_size=256, queue_size=65536

2. **Memory Bank as Implicit Hard Negatives**:
   - Queue contains representations from older batches
   - Momentum encoder ensures smooth representation space
   - Mismatch between query and momentum gives strong learning signal

3. **Momentum Encoder as Consistency Mechanism**:
   - If updating too fast: queue becomes incoherent
   - If updating too slow: can't adapt to query encoder changes
   - m=0.999 provides sweet spot (update every ~1000 batches equivalently)

4. **Single GPU Training**:
   - Unlike SimCLR which needs 8 GPUs for large batches
   - MoCo works well on single GPU
   - Effective batch size = 256, but 65536 negatives from queue

Practical Advantages:
  + Memory efficient (single GPU viable)
  + Strong negative sampling from history
  + Slower training (momentum updates) but still competitive
  + Queue provides naturally hard negatives

Practical Disadvantages:
  - More complex to implement correctly
  - Shuffling required to prevent trivial solutions
  - Momentum encoder adds overhead
  - Queue management adds complexity
"""
