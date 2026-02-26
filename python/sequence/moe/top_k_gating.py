"""
Top-K Gating Mechanism for Sparse Mixture of Experts

Advanced routing mechanism that selects top-k experts based on learned scores,
enabling sparse computation and better load balancing.

Theory:
========
Standard softmax gating: all experts contribute to output proportional to scores

    α_i = exp(gate_score_i) / Σ_j exp(gate_score_j)

Top-k gating: only top-k experts contribute (hard selection)

    selected_experts = top_k_indices(gate_scores)
    α_i = exp(gate_score_i) / Σ_j∈selected exp(gate_score_j)  if i in selected
    α_i = 0  otherwise

Advantages:
1. Sparsity: each input uses exactly k experts (predictable computation)
2. Efficiency: reduce computation to k experts instead of E
3. Load balancing: can control load across experts
4. Interpretability: clear which experts handle each input

Disadvantages:
1. Non-differentiability: argmax and selection are not differentiable
2. Gradient flow: loss of gradient signals for non-selected experts
3. Training difficulty: may require special techniques
4. Routing collapse: all examples might select same expert

Solutions:
1. Noisy Top-k Gating (Shazeer et al., 2017):
   - Add Gumbel noise during training
   - Enables exploration of all experts
   - Deterministic at inference time

2. Differentiable Top-k:
   - Use straight-through estimator
   - Relax selection to soft approximation during backward

3. Load Balancing Loss:
   - Auxiliary loss that penalizes load imbalance
   - Encourages all experts to be used

Top-k Gating with Auxiliary Loss:
=================================

Noisy Top-k Gating:

    gate_scores = W_gate @ x + b
    noisy_scores = gate_scores + Gumbel()  [only during training]
    top_k_scores, top_k_indices = top_k(noisy_scores, k)
    gate_probs = softmax(top_k_scores)

Load Balancing Loss (from Google's GShard paper):

    importance = sum(gate_probs * one_hot(top_k_indices))
    load = count(expert_selected)  [number of examples using each expert]

    balance_loss = importance / load * num_examples / batch_size
    balance_loss = mean(balance_loss)

This loss encourages:
- High importance: experts should be important (high scores)
- Even load: all experts should be used equally

Combined with main task loss:
    total_loss = task_loss + auxiliary_loss_coeff * balance_loss

Where auxiliary_loss_coeff is typically 0.01-0.1.

Mathematical Analysis:
=======================

Gumbel-Max Trick:

To sample from categorical distribution:
    sample ~ Categorical(α_1, ..., α_k)

Equivalent to:
    sample = argmax_i(log(α_i) + Gumbel_i)

where Gumbel_i ~ Gumbel(0, 1)

This enables:
1. Reparameterization: can backprop through sampling
2. During training: Gumbel noise forces exploration
3. During inference: can use deterministic max (no noise)

Gumbel Distribution:
    Gumbel(0, 1) CDF = exp(-exp(-x))
    Sampling: -log(-log(Uniform))

Benefits for MoE:
- Without noise: may always select same expert (collapse)
- With noise: different experts selected on different runs
- Forces learning of meaningful gates
- All experts contribute gradients during training

Straight-Through Estimator:

Problem: argmax is not differentiable
Solution: use approximation during backward

Forward pass:
    selected = hardmax(scores)  [hard selection]

Backward pass:
    treat as softmax (differentiable)  [approximate gradient]

This gives gradients to unselected experts, improving training.

Differentiable Top-k (Approximation):

Instead of hard selection, use soft approximation:

    α_i ≈ softmax(β * scores)

Where β is a "temperature" parameter:
- β = 1: standard softmax (all experts)
- β → ∞: approaches one-hot (single expert)

During training: use intermediate β
During inference: use high β for approximation

Load Balancing Mechanics:
=========================

Why imbalance occurs:
1. All inputs start with random gate initialization
2. Some experts get higher scores by chance
3. Those experts used more → receive more gradients
4. Those experts' gates improve → used even more
5. Other experts not used → don't improve → never used

Vicious cycle: rich get richer effect!

Load balancing loss breaks this:

    importance_i = mean(softmax(scores)_i across batch)
    load_i = number of examples with expert i in top-k
    balance_loss = variance(importance / load)

High variance = imbalance → high loss
Low variance = balanced = low loss

This encourages gradient flow to under-used experts.

Expert Diversity:

Load balancing alone not sufficient. Also need:

1. Initialization: experts initialized differently
2. Specialization: task should benefit from expert diversity
3. Diversity loss: can add loss encouraging different expert outputs

Computational Complexity:
========================

Selection cost:
- Compute gate scores: O(batch * input_dim * num_experts)
- Top-k selection: O(batch * num_experts * log(k))
- Total: O(batch * (input_dim * num_experts + num_experts * log(k)))

For num_experts = 256, input_dim = 4096:
- Gate scores: O(1M)
- Top-k: O(1K) [negligible]
- Total: dominated by gate computation

Expert computation:
- Standard: O(batch * input_dim * hidden * num_experts)
- Top-k: O(batch * input_dim * hidden * k)
- Speedup: num_experts / k

For k=4, num_experts=256: 64x speedup in expert computation!

Key Papers:
===========
1. "Outrageously Large Neural Networks for Efficient Conditional Computation"
   (Shazeer et al., 2017)
   - Original noisy top-k gating paper
   - Gumbel-max trick for exploration
   - Auxiliary loss for load balancing
   - Trained 1000+ expert models

2. "GShard: Scaling Giant Models with Conditional Computation and Automatic
   Sharding" (Lepikhin et al., 2020)
   - Production-scale MoE system
   - Improved load balancing formulation
   - Distributed training and inference

3. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and
   Efficient Sparsity" (Lewis et al., 2021)
   - Single expert selection (k=1)
   - Simplified load balancing
   - Scaled to trillion parameters
   - Cleaner training dynamics

4. "Mixture of Experts Meets Word Embeddings for Unsupervised Semantic
   Decomposition" (Lan et al., 2021)
   - Analyzes how experts specialize
   - Shows different experts learn different concepts

Implementation Considerations:
============================

Numerical Stability:

Gumbel sampling: -log(-log(uniform))
- Can have numerical issues near 0 or 1
- Use max(-log(-log(clip(uniform, eps, 1-eps))))

Load balancing loss: importance / load
- Can have division by zero if load = 0
- Use load + eps to prevent NaN

Softmax in top-k: softmax(top_k_scores)
- Can overflow if scores large
- Use: softmax(scores - max_score)

Distributed Training:

All-to-all communication:
- Gate scores computed locally
- Top-k selection per device
- Expert computation may be remote
- Results combined via reduce-scatter

Asynchronous training:
- Different devices may have different expert subsets
- Requires careful synchronization
- Can use gradual layer-wise expert allocation

Load Balancing Tuning:

Balance loss coefficient: typically 0.01-0.1
- Too low (0.001): experts don't balance, training unstable
- Too high (1.0): balancing dominates, main task suffers
- Recommendation: 0.01 for dense, 0.1 for sparse

Load balancing formula variants:
- Simple: importance / (load + eps)
- Normalized: (importance / load) / mean(importance / load)
- With threshold: only penalize large imbalances

Expert Dropout:

During training, can randomly drop experts:
- Forces network to use multiple paths
- Improves generalization
- Similar to network dropout

When to Use Top-k Gating:
========================

Use when:
- Need sparse computation (many experts)
- Have infrastructure for expert parallelization
- Model scale is large (benefits from sparsity)
- Task naturally decomposes into expert roles

Don't use when:
- Few experts (no sparsity benefit)
- Computation not bottleneck (dense forward sufficient)
- Can't afford training overhead (needs careful tuning)
- Need interpretable gradient flow
"""

from typing import Optional, Tuple
import numpy as np


def gumbel_sample(shape: Tuple) -> np.ndarray:
    """
    Sample from Gumbel(0, 1) distribution.

    Used for noisy top-k gating to enable exploration during training.

    Args:
        shape: shape of output tensor

    Returns:
        Gumbel samples
    """
    # TODO: Sample uniform random
    # uniform = np.random.uniform(0, 1, shape)

    # TODO: Ensure values not exactly 0 or 1 for numerical stability
    # uniform = np.clip(uniform, 1e-7, 1-1e-7)

    # TODO: Apply Gumbel transformation: -log(-log(u))
    # gumbel = -np.log(-np.log(uniform))

    # TODO: Return samples

    pass


class TopKGating:
    """
    Top-k gating mechanism with load balancing.

    Implements noisy top-k gating with auxiliary loss for load balancing.
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2,
                 balance_loss_coeff: float = 0.01, max_group_size: Optional[int] = None):
        """
        Initialize Top-K Gating.

        Args:
            input_dim: dimensionality of input
            num_experts: number of experts
            top_k: number of experts to select for each input
            balance_loss_coeff: coefficient for load balancing auxiliary loss
            max_group_size: maximum group size for distributed training
                           (None = no group limit)
        """
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.max_group_size = max_group_size

        # TODO: Initialize gating network parameters
        self.W_gate = None  # (input_dim, num_experts)
        self.b_gate = None  # (num_experts,)

        # TODO: Optional: noise standard deviation for Gumbel noise
        self.noise_std = None  # Learnable or fixed

    def forward(self, x: np.ndarray, training: bool = True) \
            -> Tuple[np.ndarray, np.ndarray, float, dict]:
        """
        Forward pass with top-k gating and auxiliary loss.

        Args:
            x: input, shape (batch_size, input_dim)
            training: whether in training mode (adds Gumbel noise)

        Returns:
            expert_weights: weights for selected experts (batch_size, top_k)
                          normalized softmax over selected experts
            top_k_indices: indices of selected experts (batch_size, top_k)
            aux_loss: load balancing auxiliary loss (scalar)
            gate_info: dict with additional info for analysis
        """
        batch_size = x.shape[0]

        # TODO: Compute gate logits
        # gate_logits = x @ self.W_gate + self.b_gate  # (batch, num_experts)

        # TODO: Add Gumbel noise during training for exploration
        if training:
            # gumbel_noise = gumbel_sample((batch_size, self.num_experts))
            # gate_logits = gate_logits + gumbel_noise
            pass

        # TODO: Select top-k experts
        # top_k_logits, top_k_indices = topk(gate_logits, k=self.top_k)
        # Shape: top_k_logits (batch, top_k), top_k_indices (batch, top_k)

        # TODO: Renormalize probabilities over selected experts only
        # expert_weights = softmax(top_k_logits, axis=1)

        # TODO: Compute auxiliary load balancing loss
        # This encourages even distribution of examples across experts

        # 1. Compute importance (how important each expert is)
        # importance_expert = zeros(num_experts)
        # for b in range(batch_size):
        #     for k in range(top_k):
        #         expert_idx = top_k_indices[b, k]
        #         importance_expert[expert_idx] += expert_weights[b, k]
        # importance_expert /= batch_size  # normalize

        # 2. Compute load (how many examples routed to each expert)
        # load_expert = count(expert_idx in top_k_indices for all batch and top_k)
        # load_expert /= batch_size  # normalize

        # 3. Compute balance loss (penalize imbalance)
        # balance_loss = importance_expert * load_expert
        # balance_loss = balance_loss_coeff * mean(balance_loss)

        # TODO: Store info for analysis
        # gate_info = {
        #     'gate_logits': gate_logits,
        #     'top_k_indices': top_k_indices,
        #     'expert_weights': expert_weights,
        #     'importance': importance_expert,
        #     'load': load_expert,
        # }

        # TODO: Return weights, indices, loss, and info

        pass

    def backward(self, dweights: np.ndarray, cache: dict) -> np.ndarray:
        """
        Backward pass through top-k gating.

        Args:
            dweights: gradient w.r.t. expert weights
            cache: cache from forward pass

        Returns:
            dinput: gradient w.r.t. input
        """
        # TODO: Backprop through softmax
        # TODO: Backprop through top-k selection (straight-through estimator)
        # TODO: Backprop through gate logits
        # TODO: Return gradient w.r.t. input

        pass


class LoadBalancingLoss:
    """
    Computes load balancing auxiliary loss for MoE training.

    Encourages all experts to be used equally, preventing routing collapse.
    """

    def __init__(self, num_experts: int, balance_coeff: float = 0.01,
                 normalize: bool = True):
        """
        Initialize Load Balancing Loss.

        Args:
            num_experts: number of experts
            balance_coeff: coefficient for loss weight
            normalize: whether to normalize by expected load
        """
        self.num_experts = num_experts
        self.balance_coeff = balance_coeff
        self.normalize = normalize

    def compute(self, gate_logits: np.ndarray, top_k_indices: np.ndarray,
                batch_size: Optional[int] = None) -> float:
        """
        Compute load balancing loss.

        Args:
            gate_logits: gate scores (batch_size, num_experts)
            top_k_indices: selected expert indices (batch_size, top_k)
            batch_size: total batch size (for distributed training)

        Returns:
            loss: load balancing auxiliary loss
        """
        # TODO: Compute gate probabilities via softmax
        # gate_probs = softmax(gate_logits)

        # TODO: Compute importance of each expert
        # importance = zeros(num_experts)
        # for b, batch_item in enumerate(batch):
        #     importance += gate_probs[b]
        # importance /= batch_size or len(batch)

        # TODO: Compute load (how many inputs select each expert)
        # load = zeros(num_experts)
        # for b, top_k in enumerate(top_k_indices):
        #     for expert_idx in top_k:
        #         load[expert_idx] += 1
        # load /= batch_size or len(batch)

        # TODO: Compute loss as importance * load
        # loss = sum(importance * load)
        # if self.normalize:
        #     loss = loss / num_experts

        # TODO: Scale by coefficient
        # loss = self.balance_coeff * loss

        # TODO: Return loss

        pass


class DistributedTopKGating:
    """
    Top-k gating optimized for distributed training.

    Handles all-to-all communication and load balancing across multiple devices.
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2,
                 num_devices: int = 8, balance_loss_coeff: float = 0.01):
        """
        Initialize Distributed Top-K Gating.

        Args:
            input_dim: dimensionality of input
            num_experts: total number of experts across all devices
            top_k: number of experts to select per input
            num_devices: number of devices in distributed setup
            balance_loss_coeff: auxiliary loss coefficient
        """
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_devices = num_devices
        self.balance_loss_coeff = balance_loss_coeff

        # TODO: Initialize gating parameters (same across all devices)
        self.W_gate = None  # (input_dim, num_experts)
        self.b_gate = None  # (num_experts,)

        # TODO: Device-specific expert lists
        self.experts_per_device = num_experts // num_devices

    def forward(self, x: np.ndarray, training: bool = True,
                device_id: int = 0) \
            -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward pass with distributed routing.

        Args:
            x: input from one device (batch_size, input_dim)
            training: whether in training mode
            device_id: which device this is

        Returns:
            expert_weights: weights for selected experts
            top_k_indices: global indices of selected experts
            aux_loss: load balancing loss
        """
        batch_size = x.shape[0]

        # TODO: Compute gate logits (same computation all devices)
        # gate_logits = x @ self.W_gate + self.b_gate

        # TODO: Select top-k experts (global indices)
        # top_k_logits, top_k_indices = topk(gate_logits, k=self.top_k)

        # TODO: Determine which experts are local (on this device)
        # local_expert_mask = (device_id * experts_per_device <=
        #                      top_k_indices <
        #                      (device_id + 1) * experts_per_device)

        # TODO: Need to communicate with other devices for non-local experts
        # In distributed implementation:
        # - Group examples by target expert device
        # - All-to-all communication of inputs
        # - Each device computes its local experts
        # - All-to-all communication of outputs
        # (Simplified here, actual implementation more complex)

        # TODO: Compute auxiliary loss (distributed version)
        # Need global importance and load counts

        # TODO: Return weights, indices, and loss

        pass


class SwitchGating:
    """
    Simplified gating for Switch Transformers (Lewis et al., 2021).

    Routes each input to exactly ONE expert (top_k=1) with simplified
    load balancing.
    """

    def __init__(self, input_dim: int, num_experts: int,
                 balance_loss_coeff: float = 0.01):
        """
        Initialize Switch Gating.

        Args:
            input_dim: input dimensionality
            num_experts: number of experts
            balance_loss_coeff: auxiliary loss coefficient
        """
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.balance_loss_coeff = balance_loss_coeff
        self.top_k = 1  # Always single expert

        # TODO: Initialize gating parameters
        self.W_gate = None  # (input_dim, num_experts)
        self.b_gate = None  # (num_experts,)

    def forward(self, x: np.ndarray, training: bool = True) \
            -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward pass with single expert selection.

        Args:
            x: input (batch_size, input_dim)
            training: whether in training mode

        Returns:
            expert_weights: one-hot weights for selected expert (batch_size, 1)
            expert_indices: index of selected expert (batch_size,)
            aux_loss: simplified load balancing loss
        """
        batch_size = x.shape[0]

        # TODO: Compute gate logits
        # gate_logits = x @ self.W_gate + self.b_gate

        # TODO: Add noise for exploration (optional)
        # if training:
        #     gate_logits = gate_logits + gumbel_sample(...)

        # TODO: Select top-1 (argmax)
        # expert_indices = argmax(gate_logits, axis=1)  # (batch,)

        # TODO: One-hot encoding
        # expert_weights = one_hot(expert_indices, num_experts)  # (batch, num_experts)

        # TODO: Simplified load balancing loss
        # router_z = softmax(gate_logits)  # All expert probabilities
        # importance = mean(router_z)
        # load = mean(one_hot(expert_indices))
        # balance_loss = importance * load * num_experts

        # TODO: Return weights, indices, loss

        pass


if __name__ == "__main__":
    # Test top-k gating
    batch_size, input_dim, num_experts, top_k = 32, 256, 32, 2

    # TODO: Create gating
    # gating = TopKGating(input_dim, num_experts, top_k=top_k)

    # TODO: Create sample input
    # x = np.random.randn(batch_size, input_dim)

    # TODO: Forward pass
    # weights, indices, aux_loss, info = gating.forward(x, training=True)
    # print(f"Weights shape: {weights.shape}")
    # print(f"Indices shape: {indices.shape}")
    # print(f"Auxiliary loss: {aux_loss}")
