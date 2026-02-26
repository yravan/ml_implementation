"""
Gated Recurrent Unit (GRU) Implementation

Simplified LSTM variant with 2 gates instead of 3, offering better efficiency
while maintaining strong empirical performance on many tasks.

Theory:
========
The GRU (Cho et al., 2014) simplifies LSTM by:
1. Removing the separate cell state (merges cell and hidden state)
2. Reducing gates from 3 to 2 (input+forget combined as reset+update)

This reduces parameters by ~25% while maintaining similar expressiveness.

GRU Equations:

Reset gate (controls how much of past to forget):
    r_t = sigmoid(W_r @ [x_t; h_{t-1}] + b_r)

Update gate (controls how much to update vs keep):
    z_t = sigmoid(W_z @ [x_t; h_{t-1}] + b_z)

Candidate hidden state (new information):
    h_tilde_t = tanh(W_h @ [x_t; r_t * h_{t-1}] + b_h)

Hidden state update (additive like LSTM):
    h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t

Where:
- r_t: reset gate (how much past to use)
- z_t: update gate (how much to update)
- h_tilde_t: candidate new state
- * denotes element-wise multiplication

Key Design:
===========
The update gate directly blends past and new states:
    h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t

This is equivalent to:
    h_t = z_t * h_tilde_t + (1 - z_t) * h_{t-1}

Which ensures: h_t is always a convex combination of h_{t-1} and h_tilde_t.

Gradient Flow:
==============
Like LSTM, GRU uses additive update to enable gradient flow:
    dL/dh_{t-1} = (dL/dh_t * (1 - z_t))^T (additive, not multiplicative)

For typical update gate values around 0.5:
    dL/dh_{t-1} ≈ 0.5 * dL/dh_t (much better than vanishing)

This allows information to flow backward ~100 timesteps without degradation.

Comparison with LSTM:
====================
LSTM equations:
    i_t = sigmoid(W_i @ [x_t; h_{t-1}] + b_i)
    f_t = sigmoid(W_f @ [x_t; h_{t-1}] + b_f)
    g_t = tanh(W_g @ [x_t; h_{t-1}] + b_g)
    c_t = f_t * c_{t-1} + i_t * g_t
    o_t = sigmoid(W_o @ [x_t; h_{t-1}] + b_o)
    h_t = o_t * tanh(c_t)

GRU equations:
    r_t = sigmoid(W_r @ [x_t; h_{t-1}] + b_r)
    z_t = sigmoid(W_z @ [x_t; h_{t-1}] + b_z)
    h_tilde_t = tanh(W_h @ [x_t; r_t * h_{t-1}] + b_h)
    h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t

Differences:
- LSTM has separate cell state; GRU uses hidden state directly
- LSTM has 4 gates; GRU has 2 gates
- LSTM parameters: 4(input_dim + hidden_dim + 1) * hidden_dim
- GRU parameters: 3(input_dim + hidden_dim + 1) * hidden_dim ≈ 75% of LSTM

Empirical Comparison:
- Most benchmarks: GRU ≈ LSTM (often within noise)
- Some datasets: LSTM slightly better
- Some datasets: GRU slightly better
- GRU generally preferred for efficiency/parameter count

Key Papers:
===========
1. "Learning Phrase Representations using RNN Encoder-Decoder for SMT"
   (Cho et al., 2014)
   - Original GRU paper
   - Introduced encoder-decoder architecture
   - Empirical comparison with LSTM on translation

2. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"
   (Jozefowicz et al., 2015)
   - Comprehensive GRU vs LSTM comparison
   - Shows they're often equivalent in practice
   - Task-dependent performance differences

3. "Architectures and Algorithms for Sequence Processing" (Graves et al., 2014)
   - Discusses GRU and LSTM design choices
   - Practical implementation considerations
   - Comparison with other recurrent architectures

4. "A Theoretically Grounded Application of Dropout in RNNs"
   (Gal & Ghahramani, 2016)
   - How to apply dropout to GRU
   - Variational dropout across time
   - Applies to all RNN variants

Architecture Details:
====================

1. GRU Cell (single timestep):
   - Inputs: x_t (batch, input_dim), h_t (batch, hidden_dim)
   - Outputs: h_{t+1} (batch, hidden_dim)
   - Parameters: 3 weight matrices (one per gate)
   - Total params: 3(input_dim + hidden_dim + 1) * hidden_dim
   - Example: 3 * (50 + 128 + 1) * 128 = 61,056 params per cell

2. Multi-layer GRU:
   - Stack GRUs: output of one layer → input to next
   - Each layer maintains independent hidden states
   - Popular in deep sequence models

3. Bidirectional GRU:
   - Forward GRU: processes left-to-right
   - Backward GRU: processes right-to-left
   - Concatenate outputs: (batch, seq_len, 2*hidden_dim)

4. Attention variants:
   - Attention over hidden states
   - Context vectors computed from hidden states
   - Used in machine translation

Implementation Strategy:
=======================

GRU Cell computation flow:
1. Precompute combined input: x_h = [x_t; h_{t-1}]
2. Compute reset gate: r_t = sigmoid(W_r @ x_h + b_r)
3. Compute update gate: z_t = sigmoid(W_z @ x_h + b_z)
4. Compute candidate: h_tilde = tanh(W_h @ [x_t; r_t*h_{t-1}] + b_h)
5. Update: h_{t+1} = (1-z_t)*h_t + z_t*h_tilde_t

Optimization opportunities:
- Combine reset and update gate computations before activation
- Use in-place operations where possible
- Pre-allocate hidden state buffers for sequences

Common issues:
- Careful with order of operations (reset applied before tanh)
- Bias initialization: 0 for reset/update gates
- Gradient clipping still needed even with GRU
- Variational dropout: same mask across time

Performance:
============
- Parameters: ~75% of LSTM, ~3x Vanilla RNN
- Computation: similar to LSTM, ~3x Vanilla RNN
- Memory: same asymptotic complexity
- Forward pass still sequential (cannot parallelize across time)

When to use GRU vs LSTM:
========================
Use GRU when:
- Computational efficiency is critical (fewer parameters)
- Training data is limited (fewer parameters = less overfitting)
- GPU/TPU memory is constrained
- Empirical results are similar to LSTM on your dataset

Use LSTM when:
- You have abundant training data (LSTM can use extra params)
- Computational budget is not a constraint
- Need maximum expressiveness for very long-range dependencies
- LSTM empirically outperforms on your specific task

Modern trend:
- Transformers often used instead of both GRU and LSTM
- For streaming/online settings, GRU/LSTM still preferred
- GRU becoming more common in practice due to efficiency

Variants:
=========
1. Coupled Input-Forget GRU:
   - Similar to LSTM CIFG: only 1 gate (not both r and z)
   - Further parameter reduction

2. Layer Normalized GRU:
   - Normalize activations before gates
   - Improves training stability
   - Allows higher learning rates

3. Gated Recurrent Unit with Additive Interactions:
   - Variation that uses additive interactions between reset/update
   - Sometimes performs better on specific tasks
"""

from typing import Optional, Tuple
import numpy as np


class GRUCell:
    """
    Single GRU cell implementing one timestep of computation.

    Implements the 2-gate GRU architecture:
    - Reset gate (r_t): controls past information usage
    - Update gate (z_t): controls state update amount
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize GRU cell.

        Args:
            input_dim: dimensionality of input
            hidden_dim: dimensionality of hidden state
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Combined weight matrices for reset and update gates
        # We'll use combined computation for efficiency
        # TODO: W_rz shape: (input_dim + hidden_dim, 2*hidden_dim)
        self.W_rz = None  # For reset and update gates

        # Hidden candidate weight matrix
        # TODO: W_h shape: (input_dim + hidden_dim, hidden_dim)
        self.W_h = None

        # Bias terms
        # TODO: b_rz shape: (2*hidden_dim,)
        self.b_rz = None
        # TODO: b_h shape: (hidden_dim,)
        self.b_h = None

        # Gradients
        self.dW_rz = None
        self.dW_h = None
        self.db_rz = None
        self.db_h = None

    def forward(self, x_t: np.ndarray, h_t: np.ndarray) \
            -> Tuple[np.ndarray, dict]:
        """
        Forward pass for single GRU timestep.

        Args:
            x_t: input at time t, shape (batch_size, input_dim)
            h_t: hidden state at t-1, shape (batch_size, hidden_dim)

        Returns:
            h_next: new hidden state, shape (batch_size, hidden_dim)
            cache: dict with values for backward pass
        """
        batch_size = x_t.shape[0]

        # TODO: Concatenate input and previous hidden state
        # x_h = np.concatenate([x_t, h_t], axis=1)  # (batch, input_dim + hidden_dim)

        # TODO: Compute reset and update gates
        # gates = x_h @ self.W_rz + self.b_rz  # (batch, 2*hidden_dim)
        # r_t = sigmoid(gates[:, :self.hidden_dim])
        # z_t = sigmoid(gates[:, self.hidden_dim:])

        # TODO: Compute candidate hidden state with reset gate
        # Reset gate controls how much of past hidden state to use
        # x_h_reset = np.concatenate([x_t, r_t * h_t], axis=1)
        # h_tilde = np.tanh(x_h_reset @ self.W_h + self.b_h)

        # TODO: Update hidden state
        # h_next = (1 - z_t) * h_t + z_t * h_tilde

        # TODO: Store cache for backward pass
        # cache = {
        #     'x_t': x_t,
        #     'h_t': h_t,
        #     'h_next': h_next,
        #     'r_t': r_t,
        #     'z_t': z_t,
        #     'h_tilde': h_tilde,
        # }

        pass

    def backward(self, dh_next: np.ndarray, cache: dict) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass through GRU cell.

        Args:
            dh_next: gradient w.r.t. hidden state at t+1
            cache: cache from forward pass

        Returns:
            dh: gradient w.r.t. hidden state at t
            dx: gradient w.r.t. input at t
        """
        # TODO: Extract cached values
        # x_t, h_t, h_next, r_t, z_t, h_tilde = [cache[k] for k in ...]

        # TODO: Gradient through update gate
        # dz = dh_next * (h_tilde - h_t)
        # dz_pre = dz * z_t * (1 - z_t)  # sigmoid derivative

        # TODO: Gradient through candidate hidden state
        # dh_tilde = dh_next * z_t
        # dh_tilde_pre = dh_tilde * (1 - h_tilde**2)  # tanh derivative

        # TODO: Gradient through reset gate (backprop through multiplied past state)
        # dh_tilde_combined_grad = dh_tilde_pre @ self.W_h.T
        # dr_h = dh_tilde_combined_grad[:, self.input_dim:]
        # dr = dr_h * h_t
        # dr_pre = dr * r_t * (1 - r_t)  # sigmoid derivative

        # TODO: Gradient w.r.t. previous hidden state
        # dh_t_from_z = dh_next * (1 - z_t)  # From update gate
        # dh_t_from_r = dr_h * r_t  # From reset gate
        # dh = dh_t_from_z + dh_t_from_r

        # TODO: Compute weight and input gradients
        # TODO: Return dx, dh

        pass


class GRU:
    """
    Full GRU layer processing entire sequences.

    Efficient alternative to LSTM with fewer parameters.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1, bidirectional: bool = False,
                 dropout: float = 0.0):
        """
        Initialize GRU layer.

        Args:
            input_dim: dimensionality of input sequences
            hidden_dim: dimensionality of hidden state
            output_dim: dimensionality of output
            num_layers: number of stacked GRU layers
            bidirectional: whether to use bidirectional GRU
            dropout: dropout rate between layers
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # TODO: Initialize GRU cells for each layer
        # Layer 0: input_dim -> hidden_dim
        # Layer i>0: hidden_dim -> hidden_dim
        self.cells = []

        # TODO: Initialize output projection
        self.W_hy = None  # (hidden_dim, output_dim)
        self.b_y = None   # (output_dim,)

    def forward(self, X: np.ndarray, h0: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through entire sequence.

        Args:
            X: input sequences, shape (batch_size, seq_len, input_dim)
            h0: initial hidden state, shape (batch_size, hidden_dim)

        Returns:
            outputs: all hidden states, shape (batch_size, seq_len, hidden_dim)
            h_final: final hidden state, shape (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = X.shape

        if h0 is None:
            h0 = np.zeros((batch_size, self.hidden_dim))

        # TODO: Initialize outputs array and state arrays
        # TODO: For each timestep:
        #   1. Process through all layers sequentially
        #   2. Maintain hidden state for each layer
        #   3. Store outputs for return and backprop
        # TODO: Store cache for backward pass

        pass

    def backward(self, doutputs: np.ndarray, learn_rate: float = 0.01) -> None:
        """
        Backward pass through entire sequence (BPTT).

        Args:
            doutputs: gradient w.r.t. outputs from all timesteps
            learn_rate: learning rate for weight updates
        """
        # TODO: Initialize hidden state gradients
        # TODO: Backpropagate through output projection
        # TODO: For each timestep (in reverse):
        #   1. Backpropagate through all layers
        #   2. Accumulate weight gradients
        # TODO: Apply gradient clipping (important!)
        # TODO: Update weights

        pass


class BidirectionalGRU(GRU):
    """
    Bidirectional GRU processing sequences in both directions.

    Useful for tasks where full sequence context is available
    (classification, tagging, not generation).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1, dropout: float = 0.0):
        """
        Initialize Bidirectional GRU.

        Args:
            input_dim: dimensionality of input sequences
            hidden_dim: dimensionality of each direction's hidden state
            output_dim: dimensionality of output
            num_layers: number of stacked layers
            dropout: dropout rate between layers
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers,
                         bidirectional=True, dropout=dropout)

        # TODO: Create separate forward and backward GRU cells
        self.forward_cells = []
        self.backward_cells = []

        # TODO: Output projection from 2*hidden_dim (concatenated directions)
        self.W_hy = None  # (2*hidden_dim, output_dim)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass processing sequence in both directions.

        Args:
            X: input sequences, shape (batch_size, seq_len, input_dim)

        Returns:
            outputs: concatenated bidirectional hidden states
                    shape (batch_size, seq_len, 2*hidden_dim)
        """
        # TODO: Process forward direction: x_0 -> x_1 -> ... -> x_T
        # TODO: Process backward direction: x_T -> x_{T-1} -> ... -> x_0
        # TODO: Concatenate forward and backward outputs at each position
        # TODO: Return concatenated outputs

        pass


class CoupledGRU(GRU):
    """
    Coupled GRU variant with combined reset-update gates.

    Further parameter reduction by coupling r_t and z_t gates.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1):
        """
        Initialize Coupled GRU.

        Args:
            input_dim: dimensionality of input sequences
            hidden_dim: dimensionality of hidden state
            output_dim: dimensionality of output
            num_layers: number of stacked layers
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers)

        # TODO: Only one gate instead of two (reset and update)
        # r_t = sigmoid(...)
        # z_t = 1 - r_t  (coupled)

    def forward(self, X: np.ndarray, h0: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass with coupled gates.

        Args:
            X: input sequences, shape (batch_size, seq_len, input_dim)
            h0: initial hidden state

        Returns:
            outputs: all hidden states
            h_final: final hidden state
        """
        # TODO: Similar to GRU but with z_t = 1 - r_t constraint
        # This saves one set of weights and biases

        pass


if __name__ == "__main__":
    # Test basic GRU
    batch_size, seq_len, input_dim, hidden_dim, output_dim = 32, 10, 50, 100, 10

    # TODO: Create sample data
    # X = np.random.randn(batch_size, seq_len, input_dim)

    # TODO: Create GRU model
    # gru = GRU(input_dim, hidden_dim, output_dim)
    # outputs, h_final = gru.forward(X)
    # print(f"Output shape: {outputs.shape}")
    # print(f"Final hidden shape: {h_final.shape}")

    # TODO: Test bidirectional GRU
    # bigru = BidirectionalGRU(input_dim, hidden_dim, output_dim)
    # outputs = bigru.forward(X)
    # print(f"Bidirectional output shape: {outputs.shape}")
