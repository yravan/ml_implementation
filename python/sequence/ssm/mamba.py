"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces

Modern variant of SSMs that adds input-dependent state transitions,
achieving Transformer-competitive performance with O(n) complexity.

IMPORTANT: Mamba represents a major breakthrough in sequence modeling.

Theory:
========
S4 Issue: Linear state space transitions (A doesn't depend on input)
  Problem: Can't adapt behavior based on current information
  Limitation: Same state dynamics for all inputs

Mamba Solution: Make state transitions input-dependent
  A_t = A(u_t) - state matrix depends on current input
  OR: Δ_t = Δ(u_t) - discretization step depends on input
  Result: Selective attention to relevant parts of sequence

Key Innovation: Input-Selective SSM

Instead of:
    x_t = A @ x_{t-1} + B @ u_t  [constant A]

Use:
    x_t = A(Δ_t) @ x_{t-1} + B(Δ_t) @ u_t  [Δ_t depends on u_t]

Where:
- Δ_t: input-dependent discretization step size
- A(Δ_t): state transition depends on how fast to change
- B(Δ_t): input scale depends on input importance

Why This Works:
===============

Analogy to Attention:
- Attention: for each position i, compute score α_ij for each position j
  Result: can adaptively focus on relevant positions

- Mamba: for each position t, compute Δ_t (how fast to forget past)
  Result: can adaptively decide when to update state

Key Differences from Attention:
- Attention: O(n^2) - compares all pairs
- Mamba: O(n) - each position only maintains single state

Mamba enables:
1. Adaptive dynamics: behavior changes based on input
2. Selective attention: implicitly focus on important information
3. O(n) complexity: still linear time and space
4. Streaming capable: process one token at a time

Mathematical Formulation:
========================

Parameterization:
    u_t ∈ ℝ^D: input at time t
    x_t ∈ ℝ^N: state (hidden)
    y_t ∈ ℝ^D: output

State-space model:
    ẋ(t) = A x(t) + B u(t)
    y(t) = C x(t) + D u(t)

Key difference in Mamba: discretization step depends on input

    Δ_t = Δ(u_t)  [learned projection of u_t]
    A_t = exp(Δ_t A)  [exponential discretization]
    B_t = Δ_t B  [scale input by Δ_t]

Discrete update:
    x_t = A_t x_{t-1} + B_t u_t

This is much simpler than S4's structured matrices but more flexible
because Δ_t can vary per position.

Selective Mechanism:
====================

The Δ_t parameter acts as a learned "selection" mechanism:

1. Δ_t large: large step size
   → faster state updates
   → "reset" state more aggressively
   → forget past information quickly

2. Δ_t small: small step size
   → slower state updates
   → preserve past information
   → accumulate long-term context

This enables the model to:
- Input from "important" tokens: use large Δ (reset state)
- Skip "unimportant" tokens: use small Δ (preserve state)
- Implicit selection without explicit attention

Architecture: ScanLayer
=======================

The Mamba block differs from standard SSMs with:

1. Projection layers:
   u_t = W_in * input_t  [project input]

2. Gating mechanism:
   z_t = σ(W_z * u_t)  [learned gate]

3. SSM with selective step:
   Δ_t = softplus(W_Δ * u_t) + bias  [input-dependent discretization]
   x_t = A(Δ_t) x_{t-1} + B(Δ_t) u_t

4. Gate application:
   output = z_t ⊙ SSM_out_t

5. Projection:
   y_t = W_out * output  [project to output dimension]

Why Selective is Important:
===========================

Copy Task: Mamba vs S4
- Input: sequence of random tokens + target token + padding
- Goal: output the target token after padding
- S4 struggles: same A for all inputs, can't "decide" when to copy
- Mamba succeeds: Δ_t large when seeing target, small when padding
  Result: state updated with target, preserved through padding

This is the "induction head" problem from attention literature.
Mamba solves it with input-dependent selectivity!

Efficiency Advantages:
======================

Time Complexity:
- Standard Attention: O(n^2)
- Linear Attention: O(n) [approximate, may lose expressiveness]
- Mamba: O(n) [exact, with selectivity]

Memory:
- Attention: O(n^2) [store attention matrix]
- Linear Attention: O(n)
- Mamba: O(n)

Computation per step:
- Can use efficient scan operation (parallel prefix sum)
- Better parallelization than RNN (not fully sequential)
- Not as parallel as Attention, but very fast in practice

Hardware Efficiency:
- Mamba uses GPU-friendly operations
- Optimized CUDA kernels available
- Better hardware utilization than attention for long sequences

Key Papers:
===========
1. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
   (Gu & Dao, 2023) - The original paper
   - Introduces selective state spaces
   - Achieves Transformer-competitive performance
   - O(n) complexity with selectivity
   - Shows empirical SOTA on various tasks

2. Builds on: "Efficiently Modeling Long Sequences with Structured State Spaces"
   (Gu et al., 2021) - S4 paper
   - Foundation: structured SSMs
   - Kernel trick for efficiency

3. Related: "Selective Attention Networks" (various papers)
   - Attention with gating/selection
   - Can be viewed as attention analogue of Mamba

Empirical Results:
==================

Mamba Performance:
- Language modeling: competitive with Transformers
- Long-range tasks: better than Transformers
- Downstream tasks: strong when fine-tuned
- Training speed: faster than Transformers on long sequences
- Inference latency: much faster than Transformers

Specific Benchmarks (from paper):
- 1B parameters: competitive with 3B Transformer
- Long Document Understanding: 8x faster, better accuracy
- ImageNet (visual): competitive with ViT
- Speech: strong results on audio tasks

Breakthrough: For the first time, SSM-based model competitive with
Transformers on general NLP tasks, not just specialized long-seq tasks!

Architecture Variants:
=====================

1. Bidirectional Mamba:
   - Process sequence forward and backward
   - Concatenate representations
   - Good for tasks with full context (classification, tagging)

2. Mamba with Local Sliding Window:
   - Combine Mamba with local attention
   - Local: high expressiveness
   - Global (via Mamba): efficient long-range

3. Multi-Head Mamba:
   - Multiple selection mechanisms
   - Different Δ for different "heads"
   - Richer representations

4. Hierarchical Mamba:
   - Different layer types: some local, some global
   - Lower layers: local for fine details
   - Upper layers: global/Mamba for structure

Implementation Considerations:
=============================

Parameter Count:
- W_in: D_in × D_inner
- W_Δ: D_inner × rank (rank << D_inner for efficiency)
- W_z: D_inner × D_inner (gating)
- W_out: D_inner × D_out
- Total: similar to single Transformer layer!

Initialization:
- Important for stability
- Δ bias: initialize to small positive values
- A: initialize near -1 to -0.5 (stable, not too fast)
- Others: standard initializations

Numerical Stability:
- Exp(Δ_t A) can overflow/underflow
- Use log-space computation where possible
- Careful with softplus function

Training Considerations:
- Selective mechanism can be hard to learn
- May need curriculum or careful initialization
- Gradient flow: state-based, different from attention
- Can use gradient checkpointing for memory efficiency

Parallelization:
- Sequential RNN-style: O(n) serial
- Parallel via scan: O(log n) depth with O(n) processors
- Not as parallel as full attention, but much better than RNN

Streaming/Online:
- Natural for streaming: maintain state, process one token at a time
- Δ learned during training works for streaming
- Good for real-time applications

When to Use Mamba:
==================

Use Mamba when:
1. Sequence length very long (10K+ tokens)
   - O(n) vs O(n^2) for attention
   - Much faster training and inference

2. Need streaming/online processing
   - Process tokens as they arrive
   - Maintain compact state

3. Model size critical
   - Smaller than equivalent Transformers
   - Better parameter efficiency

4. Inference latency important
   - Much faster than Transformers
   - Token-at-a-time processing

5. Training efficiency matters
   - Faster to train than Transformers on long sequences
   - Less memory required

Use Transformers when:
1. Sequence length moderate (100-2000)
   - Attention has better constants
   - More research/optimization

2. Need maximum interpretability
   - Attention weights clear
   - State transitions less clear

3. Have unlimited compute budget
   - Transformers well-optimized
   - More variants and techniques

4. Task specifically requires flexible routing
   - Some tasks benefit from explicit attention
   - Sentiment analysis, machine translation

Current State:
==============
- Mamba very recent (2023)
- Still being researched and improved
- Already showing promise
- Expected to influence sequence modeling for years

Trend: Mamba likely to become dominant for:
- Long sequences (>1000)
- Streaming/online applications
- Resource-constrained environments

Transformers likely to remain strong for:
- Moderate sequences
- NLP with specialized pre-training
- Multi-modal learning (vision + language)

Future: Likely convergence of best ideas from both paradigms!
"""

from typing import Optional, Tuple
import numpy as np


class MambaBlock:
    """
    Mamba selective state space block.

    Implements input-dependent state transitions for adaptive sequence modeling.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand_ratio: int = 2):
        """
        Initialize Mamba Block.

        Args:
            d_model: model dimensionality (input/output)
            d_state: state dimension (N in theory)
            d_conv: local convolution dimension
            expand_ratio: hidden layer expansion (d_inner = expand * d_model)
        """
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand_ratio * d_model

        # TODO: Initialize projections
        self.W_in = None  # (d_model, d_inner) - expand input

        # TODO: Selection mechanism (input-dependent Δ)
        # Δ_t = softplus(W_Δ @ u_t + b_Δ)
        self.W_delta = None  # (d_inner, d_state) - compute Δ
        self.b_delta = None  # (d_state,) bias for Δ

        # TODO: SSM parameters (learned, but not input-dependent)
        # A is stable matrix, typically diagonal
        self.A = None  # (d_state,) diagonal entries
        self.B = None  # (d_state, 1) or similar
        self.C = None  # (1, d_state) or similar

        # TODO: Gating mechanism
        self.W_z = None  # (d_inner, d_inner) - compute gate

        # TODO: Output projection
        self.W_out = None  # (d_inner, d_model) - project back

        # TODO: Local mixing (optional, for expressiveness)
        self.conv_kernel = None  # (d_conv,) for local filtering

    def forward(self, u: np.ndarray, state: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through Mamba block.

        Args:
            u: input sequence, shape (batch_size, seq_len, d_model)
            state: initial state, shape (batch_size, d_state)

        Returns:
            y: output, shape (batch_size, seq_len, d_model)
            state_final: final state, shape (batch_size, d_state)
        """
        batch_size, seq_len, _ = u.shape

        if state is None:
            state = np.zeros((batch_size, self.d_state))

        # TODO: Expand input
        # u_expanded = u @ self.W_in  # (batch, seq_len, d_inner)

        # TODO: Compute input-dependent discretization step
        # delta_t = softplus(u_expanded @ self.W_delta + self.b_delta)
        # shape: (batch, seq_len, d_state)

        # TODO: Discretize A and B using Δ_t
        # A_d = exp(Δ_t * A)  [element-wise for diagonal A]
        # B_d = Δ_t * B

        # TODO: SSM forward pass (selective state transitions)
        # x_t = A_d[t] * x_{t-1} + B_d[t] * u_t
        # y_t = C @ x_t

        # TODO: Apply gating
        # z_t = sigmoid(u_expanded @ self.W_z)
        # y_gated = z_t * y_t

        # TODO: Project output
        # y_proj = y_gated @ self.W_out

        # TODO: Return output and final state

        pass

    def backward(self, dy: np.ndarray, cache: dict) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass through Mamba block.

        Args:
            dy: gradient w.r.t. output
            cache: cache from forward pass

        Returns:
            du: gradient w.r.t. input
            dstate: gradient w.r.t. initial state
        """
        # TODO: Backward through output projection
        # TODO: Backward through gating
        # TODO: Backward through SSM (selective state backward)
        # TODO: Backward through discretization
        # TODO: Backward through input expansion
        # TODO: Return gradients

        pass


class BiDirectionalMamba(MambaBlock):
    """
    Bidirectional Mamba for tasks with full context (classification, tagging).

    Processes sequence forward and backward, concatenates representations.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand_ratio: int = 2):
        """
        Initialize Bidirectional Mamba.

        Args:
            d_model: model dimensionality
            d_state: state dimension
            d_conv: convolution dimension
            expand_ratio: expansion ratio
        """
        super().__init__(d_model, d_state, d_conv, expand_ratio)

        # TODO: Create separate forward and backward Mamba blocks
        self.forward_block = MambaBlock(d_model, d_state, d_conv, expand_ratio)
        self.backward_block = MambaBlock(d_model, d_state, d_conv, expand_ratio)

        # TODO: Projection to combine forward and backward
        self.W_combine = None  # (2*d_inner, d_model)

    def forward(self, u: np.ndarray) -> np.ndarray:
        """
        Forward pass processing in both directions.

        Args:
            u: input sequence (batch_size, seq_len, d_model)

        Returns:
            y: output (batch_size, seq_len, d_model)
        """
        # TODO: Process forward direction
        # y_forward, _ = self.forward_block.forward(u)

        # TODO: Reverse sequence for backward direction
        # u_reversed = reverse(u)
        # y_backward, _ = self.backward_block.forward(u_reversed)
        # y_backward = reverse(y_backward)

        # TODO: Concatenate forward and backward
        # combined = concatenate([y_forward, y_backward], axis=-1)

        # TODO: Project to output dimension
        # y = combined @ self.W_combine

        # TODO: Return output

        pass


class Mamba:
    """
    Full Mamba model with multiple selective state space layers.

    Can be used as backbone for various sequence-to-sequence tasks.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int,
                 num_layers: int, expand_ratio: int = 2,
                 vocab_size: Optional[int] = None,
                 num_classes: Optional[int] = None):
        """
        Initialize Mamba model.

        Args:
            d_model: model dimensionality
            d_state: state dimension for each Mamba block
            d_conv: convolution dimension
            num_layers: number of Mamba blocks to stack
            expand_ratio: hidden layer expansion ratio
            vocab_size: if provided, add embedding and language head
            num_classes: if provided, add classification head
        """
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_layers = num_layers
        self.expand_ratio = expand_ratio

        # TODO: Initialize embedding layer (optional)
        if vocab_size:
            self.embedding = None  # (vocab_size, d_model)
        else:
            self.embedding = None

        # TODO: Initialize Mamba layers
        self.layers = []
        for _ in range(num_layers):
            layer = MambaBlock(d_model, d_state, d_conv, expand_ratio)
            self.layers.append(layer)

        # TODO: Initialize language modeling head (optional)
        if vocab_size:
            self.W_lm = None  # (d_model, vocab_size)

        # TODO: Initialize classification head (optional)
        if num_classes:
            self.W_cls = None  # (d_model, num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through Mamba model.

        Args:
            x: input (batch_size, seq_len) if embedding enabled
               or (batch_size, seq_len, d_model) if raw embeddings

        Returns:
            y: output (batch_size, seq_len, d_model) or class logits
        """
        batch_size, seq_len = x.shape[:2]

        # TODO: Embed if needed
        # TODO: Initialize states for all layers

        # TODO: Forward through each Mamba layer
        # Use residual connections between layers

        # TODO: Apply output head (language modeling or classification)

        # TODO: Return output

        pass

    def forward_for_language_modeling(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass for language modeling task.

        Args:
            input_ids: token IDs (batch_size, seq_len)

        Returns:
            logits: vocabulary logits (batch_size, seq_len, vocab_size)
        """
        # TODO: Embed input tokens
        # TODO: Forward through Mamba layers
        # TODO: Project to vocabulary logits
        # TODO: Return logits

        pass

    def forward_for_classification(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass for sequence classification.

        Args:
            input_ids: token IDs (batch_size, seq_len)

        Returns:
            logits: class logits (batch_size, num_classes)
        """
        # TODO: Embed input tokens
        # TODO: Forward through Mamba layers
        # TODO: Extract final token representation (or pool)
        # TODO: Project to class logits
        # TODO: Return logits

        pass

    def forward_streaming(self, token_id: int, state: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for streaming/online processing.

        Process one token at a time, maintaining state.
        Enables real-time applications.

        Args:
            token_id: current token ID
            state: state from previous timesteps (list of arrays per layer)

        Returns:
            logits: predictions for next token (vocab_size,)
            state_new: updated state for next call
        """
        # TODO: Embed single token
        # TODO: For each layer:
        #   1. Update state: x_t = A_d(Δ_t) @ x_{t-1} + B_d(Δ_t) @ u_t
        #   2. Compute output
        #   3. Update state tracking

        # TODO: Project to logits
        # TODO: Return logits and new state

        pass


class MambaConfig:
    """
    Configuration for Mamba model.

    Simplifies model creation with standard configurations.
    """

    def __init__(self, d_model: int = 256, d_state: int = 16,
                 d_conv: int = 4, num_layers: int = 24,
                 expand_ratio: int = 2):
        """
        Initialize Mamba configuration.

        Args:
            d_model: model dimension
            d_state: state dimension
            d_conv: convolution kernel size
            num_layers: number of Mamba blocks
            expand_ratio: hidden expansion
        """
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_layers = num_layers
        self.expand_ratio = expand_ratio

    @classmethod
    def small(cls):
        """Small Mamba: ~125M parameters."""
        return cls(d_model=768, num_layers=24)

    @classmethod
    def base(cls):
        """Base Mamba: ~370M parameters."""
        return cls(d_model=1024, num_layers=24)

    @classmethod
    def large(cls):
        """Large Mamba: ~1.3B parameters."""
        return cls(d_model=1536, num_layers=48)


if __name__ == "__main__":
    # Test Mamba block
    batch_size, seq_len, d_model = 4, 100, 256

    # TODO: Create Mamba block
    # block = MambaBlock(d_model)

    # TODO: Create sample input
    # u = np.random.randn(batch_size, seq_len, d_model)

    # TODO: Forward pass
    # y, state = block.forward(u)
    # print(f"Output shape: {y.shape}")
    # print(f"State shape: {state.shape}")

    # TODO: Test full Mamba model
    # config = MambaConfig.small()
    # mamba = Mamba(config.d_model, config.d_state, config.d_conv,
    #              config.num_layers, vocab_size=50000)
