"""
Mixture of Experts (MoE) Layer Implementation

Enables conditional computation where different experts specialize on different
input patterns. Dramatically scales model capacity without proportional increase
in computation cost.

Theory:
========
The core idea of MoE is to decompose the model into multiple "experts" and use
a learned "routing" or "gating" network to select which experts to use for each
input.

Standard Feed-Forward Network:
    output = W_out * relu(W_in * x + b_in) + b_out
    Parameters: (input_dim → hidden_dim → output_dim)
    Computation: always passes through all weights

Mixture of Experts:
    expert_i(x) = W_out_i * relu(W_in_i * x + b_in_i) + b_out_i  for i = 1..E
    g(x) = softmax(W_gate * x)  [learned routing probabilities]
    output = Σ_i g_i(x) * expert_i(x)

Where:
- E: number of experts
- g_i(x): gating probability (how much to use expert i)
- expert_i(x): output of expert i
- output: weighted combination of all experts

Key Insight: Different experts specialize on different input types
- Expert 1: might learn patterns for class A
- Expert 2: might learn patterns for class B
- Expert 3: might learn patterns for class C
- Gating network learns which expert to use for each input

Advantages:
1. Conditional Computation: only use relevant expert(s) for each input
   - Reduces computation per input
   - Can have many experts without hurting latency

2. Increased Model Capacity:
   - Total parameters: E * (input_dim → hidden_dim → output_dim)
   - But computation per forward pass: only uses subset of experts
   - Can add experts without increasing inference cost much

3. Dynamic Routing:
   - Model learns specialization automatically
   - Different experts learn different skills
   - Can be adaptive to input type

Example: Model with 256 experts
- Total parameters: ~4B (if expert hidden_dim = 4096)
- Computation per input: only ~64M if using 4 experts on average
- This allows 10-50x parameter scaling with <2x computation

Hard vs Soft Routing:
====================

Soft Routing (standard MoE):
    output = Σ_i softmax(gate_scores)_i * expert_i(x)
    All experts contribute to output
    Differentiable everywhere (easier training)

Hard Routing:
    expert_i selected if top_k(gate_scores)
    output = Σ_i∈top_k softmax(selected_scores)_i * expert_i(x)
    Only top-k experts used
    More sparsity but harder to train (gradient through selection)

Gating Mechanisms:
==================

1. Standard Softmax Gate:
    g_i(x) = exp(w_i · x) / Σ_j exp(w_j · x)

   Simple, soft routing
   All experts contribute, weighted by scores

2. Top-k Gating:
    Selected: top_k indices of scores
    g_i(x) = exp(w_i · x) / Σ_j∈top_k exp(w_j · x)

   Hard routing: only top-k experts used
   More efficient but more discreteness

3. Noisy Top-k (Shazeer et al., 2017):
    gate_scores = w · x + Gumbel_noise
    Selected: top_k indices of noisy scores

   Adds stochasticity during training
   Forces exploration of all experts
   But deterministic at test time

4. Load Balancing Gate:
    Combines top-k with auxiliary loss to balance load:

    auxiliary_loss = Σ_i (fraction_used_i * expert_load_i)

   Prevents collapse where all examples route to same expert
   Encourages load distribution across experts

Scaling and Efficiency:
=======================

Expert Parallelization:
- Each expert can be on different device (data parallelism)
- Requires communication of gate outputs before expert computation
- All-to-all communication step

Model Parallelism:
- Experts distributed across devices
- Input duplicated, output reduce-scatter
- Can use different communication patterns (ring, mesh, etc.)

Computation:
- Gate network: O(batch_size * input_dim * num_experts) [very fast]
- Expert computation: O(batch_size * input_dim * hidden_dim * active_experts)
- Routing: O(batch_size * num_experts) for sorting/selection

Load Balancing Challenge:
- If gate learns to use only subset of experts, rest don't train
- Solution: auxiliary loss penalizing load imbalance
- Solution: noisy routing adds exploration
- Solution: expert randomization during training

Key Papers:
===========
1. "Outrageously Large Neural Networks for Efficient Conditional Computation"
   (Shazeer et al., 2017)
   - Original MoE scaling work
   - Noisy top-k gating
   - Balancing loads with auxiliary loss
   - Showed 1000+ expert training

2. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and
   Efficient Sparsity" (Lewis et al., 2021)
   - Simplified MoE (single expert selection)
   - Scales to trillion parameters
   - Simplified load balancing
   - Better training stability

3. "GShard: Scaling Giant Models with Conditional Computation and Automatic
   Sharding" (Lepikhin et al., 2020)
   - Production MoE system
   - Distributed training of large models
   - Communication optimization
   - Load balancing improvements

4. "Beyond Distillation: Task-level Mixture of Experts"
   (Kudugunta et al., 2021)
   - MoE for multi-task learning
   - Task-specific routing
   - Shared base model with expert selection per task

5. "Adaptive Mixture of Experts with Soft Tree Gating" (Ma et al., 2021)
   - Hierarchical MoE (tree structure)
   - Better routing mechanism
   - Improved training stability

Architecture Details:
====================

1. Expert Design:
   - Often: simple feed-forward networks
   - Can share parts (e.g., input projection) for efficiency
   - Experts need diversity to specialize

2. Gating Network:
   - Simple linear layer: g = softmax(W @ x)
   - Possibly with temperature scaling
   - Can be very efficient (minimal overhead)

3. Load Balancing:
   - Auxiliary loss: importance * load * coefficient
   - Important to prevent expert collapse
   - Balance coefficient: typically 0.01-0.1

4. Communication Pattern:
   - All-to-all or collective operations
   - Can be bottleneck for distributed training
   - Depends on expert placement and batch size

Implementation Strategy:
=======================

Basic MoE Layer Structure:
```python
class MoELayer:
    def forward(self, x):
        # Gate computation
        gate_scores = self.gate(x)  # (batch, num_experts)
        gate_probs = softmax(gate_scores)

        # Expert computation
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # (batch, output_dim)
            expert_outputs.append(expert_out)

        # Weighted combination
        outputs = []
        for i, out in enumerate(expert_outputs):
            weighted = gate_probs[:, i:i+1] * out
            outputs.append(weighted)

        output = sum(outputs)
        return output, gate_probs
```

For Top-k Gating:
```python
def forward(self, x):
    gate_scores = self.gate(x)  # (batch, num_experts)

    # Select top-k
    top_k_scores, top_k_indices = topk(gate_scores, k=self.k)
    gate_probs = softmax(top_k_scores)  # Only top-k

    # Only compute top-k experts
    expert_outputs = self.experts[top_k_indices](x)

    # Weighted combination
    output = (gate_probs * expert_outputs).sum(dim=1)
    return output, gate_probs
```

Common Issues:
- Load imbalance: auxiliary loss necessary
- Gradient flow: noisy routing helps
- Communication overhead: need efficient distributed implementation
- Training instability: careful learning rate scheduling needed

Comparison with Other Approaches:
==================================

MoE vs Deep Networks:
- MoE: more parameters, same computation
- Deep: more computation per forward pass
- MoE: better for conditional specialization
- Deep: better for sequential reasoning

MoE vs Ensemble:
- MoE: single model with learned routing
- Ensemble: multiple independent models
- MoE: more efficient (shared base, single forward pass)
- Ensemble: fully independent but more computation

MoE vs Attention:
- MoE: position-agnostic routing (doesn't depend on positions)
- Attention: learns which positions to attend to
- MoE: good for input-type-dependent specialization
- Attention: good for sequence-level dependencies

When to Use:
============
Use MoE when:
- Need to scale model without proportional compute increase
- Input types are diverse (natural specialization)
- Have distributed training setup
- Inference latency is critical

Don't use MoE when:
- Model size not a bottleneck
- Inference speed less important than throughput
- Can't afford communication overhead
- Prefer simpler architecture
"""

from typing import Optional, Tuple, List
import numpy as np


class Expert:
    """
    Individual expert module in Mixture of Experts.

    Typically a simple feed-forward network that specializes on certain patterns.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize Expert.

        Args:
            input_dim: input dimensionality
            hidden_dim: hidden layer size
            output_dim: output dimensionality
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # TODO: Initialize parameters
        self.W_in = None   # (input_dim, hidden_dim)
        self.b_in = None   # (hidden_dim,)
        self.W_out = None  # (hidden_dim, output_dim)
        self.b_out = None  # (output_dim,)

        # Gradients
        self.dW_in = None
        self.db_in = None
        self.dW_out = None
        self.db_out = None

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass through expert.

        Args:
            x: input, shape (batch_size, input_dim)

        Returns:
            output: expert output, shape (batch_size, output_dim)
            cache: for backward pass
        """
        # TODO: Implement forward pass
        # hidden = relu(x @ self.W_in + self.b_in)
        # output = hidden @ self.W_out + self.b_out
        # TODO: Cache for backward
        # TODO: Return output and cache

        pass

    def backward(self, doutput: np.ndarray, cache: dict, learn_rate: float = 0.01) -> np.ndarray:
        """
        Backward pass through expert.

        Args:
            doutput: gradient w.r.t. output
            cache: cache from forward pass
            learn_rate: learning rate for weight updates

        Returns:
            dinput: gradient w.r.t. input
        """
        # TODO: Implement backward pass
        # TODO: Compute gradients w.r.t. weights
        # TODO: Update weights
        # TODO: Return gradient w.r.t. input

        pass


class GatingNetwork:
    """
    Learned routing network that selects which experts to use.

    Computes a probability distribution over experts based on input.
    """

    def __init__(self, input_dim: int, num_experts: int):
        """
        Initialize Gating Network.

        Args:
            input_dim: dimensionality of input
            num_experts: number of experts to route to
        """
        self.input_dim = input_dim
        self.num_experts = num_experts

        # TODO: Initialize gating parameters (simple linear layer)
        self.W_gate = None  # (input_dim, num_experts)
        self.b_gate = None  # (num_experts,)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Compute gating probabilities.

        Args:
            x: input, shape (batch_size, input_dim)

        Returns:
            gate_probs: softmax probabilities, shape (batch_size, num_experts)
            cache: for backward pass
        """
        # TODO: Compute gate scores: x @ W_gate + b_gate
        # TODO: Apply softmax to get probabilities
        # TODO: Cache for backward
        # TODO: Return probabilities and cache

        pass

    def backward(self, dgate_probs: np.ndarray, cache: dict,
                 learn_rate: float = 0.01) -> np.ndarray:
        """
        Backward pass through gating network.

        Args:
            dgate_probs: gradient w.r.t. gate probabilities
            cache: cache from forward pass
            learn_rate: learning rate

        Returns:
            dinput: gradient w.r.t. input
        """
        # TODO: Implement backward through softmax and linear layer
        # TODO: Update gating parameters
        # TODO: Return gradient w.r.t. input

        pass


class MoELayer:
    """
    Mixture of Experts layer with learned routing.

    Enables conditional computation where different experts specialize on
    different input types.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_experts: int, expert_hidden_dim: Optional[int] = None,
                 top_k: int = 1, balance_loss_coeff: float = 0.01):
        """
        Initialize Mixture of Experts Layer.

        Args:
            input_dim: input dimensionality
            hidden_dim: hidden layer size (not used in pure MoE, for compatibility)
            output_dim: output dimensionality
            num_experts: number of expert modules
            expert_hidden_dim: hidden dimension of each expert
                              (default: same as hidden_dim)
            top_k: number of experts to select (1 for Switch, >1 for full MoE)
            balance_loss_coeff: coefficient for load balancing auxiliary loss
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim or hidden_dim
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff

        # TODO: Initialize gating network
        self.gating = GatingNetwork(input_dim, num_experts)

        # TODO: Initialize all experts
        # Expert can output to hidden_dim or directly to output_dim
        # Common: experts output to hidden_dim, then project to output_dim
        self.experts = []
        for _ in range(num_experts):
            expert = Expert(input_dim, self.expert_hidden_dim, output_dim)
            self.experts.append(expert)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward pass through MoE layer.

        Args:
            x: input, shape (batch_size, input_dim)

        Returns:
            output: MoE output, shape (batch_size, output_dim)
            expert_weights: weights assigned to each expert (batch_size, num_experts)
            balance_loss: auxiliary loss to balance expert usage
        """
        batch_size = x.shape[0]

        # TODO: Compute gating probabilities
        # gate_probs, gate_cache = self.gating.forward(x)  # (batch, num_experts)

        # TODO: Select top-k experts if using sparse routing
        if self.top_k == 1:
            # Switch routing: select single best expert
            # expert_indices = argmax(gate_probs)
            pass
        else:
            # Top-k routing: select top-k experts
            # top_k_indices = topk(gate_probs, k=self.top_k)
            pass

        # TODO: Compute outputs from all experts
        # expert_outputs = []
        # for expert in self.experts:
        #     out, cache = expert.forward(x)
        #     expert_outputs.append(out)

        # TODO: Combine expert outputs using gate weights
        # output = sum(gate_probs[:, i:i+1] * expert_outputs[i] for i in range(num_experts))

        # TODO: Compute load balancing auxiliary loss
        # This encourages equal usage of all experts
        # importance = sum(gate_probs)  # how important each expert is
        # load = count(expert_used)  # how much each expert is used
        # balance_loss = balance_loss_coeff * sum(importance * load / batch_size)

        # TODO: Cache for backward pass

        # TODO: Return output, gate probabilities, and balance loss

        pass

    def backward(self, doutput: np.ndarray, learn_rate: float = 0.01) -> np.ndarray:
        """
        Backward pass through MoE layer.

        Args:
            doutput: gradient w.r.t. output
            learn_rate: learning rate

        Returns:
            dinput: gradient w.r.t. input
        """
        # TODO: Backprop through weighted sum (experts)
        # For each expert: d_expert_out = d_output * gate_prob
        # TODO: Backprop through each expert
        # TODO: Accumulate gradients of gate with respect to gating network
        # TODO: Backprop through gating network
        # TODO: Combine input gradients from all experts
        # TODO: Return gradient w.r.t. input

        pass


class TopKGatingMoE(MoELayer):
    """
    MoE with top-k sparse gating (Shazeer et al., 2017).

    Only the top-k experts are used for each input, reducing computation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_experts: int, expert_hidden_dim: Optional[int] = None,
                 top_k: int = 2, use_noisy_gating: bool = True):
        """
        Initialize Top-K MoE.

        Args:
            input_dim: input dimensionality
            hidden_dim: hidden dimension
            output_dim: output dimensionality
            num_experts: number of experts
            expert_hidden_dim: expert hidden dimension
            top_k: number of top experts to select
            use_noisy_gating: whether to add noise for exploration
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_experts,
                        expert_hidden_dim, top_k=top_k)

        self.use_noisy_gating = use_noisy_gating

        # TODO: If using noisy gating, initialize noise standard deviation
        if use_noisy_gating:
            self.noise_std = None  # (num_experts,) trainable parameter

    def forward(self, x: np.ndarray, training: bool = True) \
            -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward pass with top-k sparse gating.

        Args:
            x: input, shape (batch_size, input_dim)
            training: whether in training mode (adds noise) or inference (deterministic)

        Returns:
            output: sparse MoE output
            expert_weights: weights of selected experts
            balance_loss: auxiliary loss
        """
        batch_size = x.shape[0]

        # TODO: Compute gating scores
        # gate_scores = x @ self.gating.W_gate + self.gating.b_gate

        # TODO: Add Gumbel noise for exploration during training
        if training and self.use_noisy_gating:
            # gate_scores += Gumbel_sample()

            pass

        # TODO: Select top-k experts
        # top_k_logits, top_k_indices = topk(gate_scores, k=self.top_k)

        # TODO: Renormalize probabilities over selected experts only
        # top_k_gates = softmax(top_k_logits)

        # TODO: Only compute forward for selected experts
        # sparse_output = zeros((batch_size, output_dim))
        # for i, expert_idx in enumerate(top_k_indices):
        #     expert_output = self.experts[expert_idx].forward(x)
        #     sparse_output += top_k_gates[:, i:i+1] * expert_output

        # TODO: Compute auxiliary load balancing loss
        # importance = sum(top_k_gates)
        # load = count(expert_selected)
        # balance_loss = ...

        # TODO: Return output and loss

        pass


class SwitchTransformer:
    """
    Switch Transformer variant of MoE (Lewis et al., 2021).

    Simplified MoE where each token is routed to exactly ONE expert.
    Simplification enables better scalability (trillion parameter models).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_experts: int, expert_hidden_dim: Optional[int] = None):
        """
        Initialize Switch Transformer.

        Args:
            input_dim: input dimensionality
            hidden_dim: hidden dimension
            output_dim: output dimensionality
            num_experts: number of experts
            expert_hidden_dim: expert hidden dimension
        """
        # Use MoE with top_k=1 (single expert per input)
        super().__init__(input_dim, hidden_dim, output_dim, num_experts,
                        expert_hidden_dim, top_k=1)

        # TODO: Initialize simplified load balancing
        # Switch uses different load balancing strategy

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward pass routing each input to single expert.

        Args:
            x: input, shape (batch_size, input_dim)

        Returns:
            output: Switch MoE output
            expert_assignment: which expert handles each input
            load_balance_loss: simplified load balancing loss
        """
        # TODO: Implement single-expert routing
        # gate_scores = x @ self.gating.W_gate
        # expert_indices = argmax(gate_scores)
        # gate_probs = one_hot(expert_indices)

        # TODO: Route to single expert
        # output = experts[expert_indices](x)

        # TODO: Compute load balancing loss
        # balance_loss = auxiliary_loss(expert_indices)

        pass


if __name__ == "__main__":
    # Test MoE layer
    batch_size, input_dim, hidden_dim, output_dim, num_experts = 32, 256, 1024, 256, 32

    # TODO: Create MoE layer
    # moe = MoELayer(input_dim, hidden_dim, output_dim, num_experts)

    # TODO: Create sample input
    # x = np.random.randn(batch_size, input_dim)

    # TODO: Forward pass
    # output, gate_probs, balance_loss = moe.forward(x)
    # print(f"Output shape: {output.shape}")
    # print(f"Gate shape: {gate_probs.shape}")
    # print(f"Balance loss: {balance_loss}")
