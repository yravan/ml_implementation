"""
Long Short-Term Memory (LSTM) Network Implementation

Comprehensive implementation of LSTM cells and layers that overcome vanishing
gradient problems through gating mechanisms and cell state.

Theory:
========
The LSTM introduces a "cell state" c_t that runs alongside the hidden state h_t.
This allows the network to maintain information over long time horizons through
multiplicative interactions (gates).

Core Equations (Hochreiter & Schmidhuber, 1997):

Input gate (controls what new information enters):
    i_t = sigmoid(W_ii @ x_t + W_hi @ h_{t-1} + b_i)

Forget gate (controls what information to discard):
    f_t = sigmoid(W_if @ x_t + W_hf @ h_{t-1} + b_f)

Cell state candidate (new information to potentially add):
    g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)

Cell state update (multiplicative interaction):
    c_t = f_t * c_{t-1} + i_t * g_t

Output gate (controls what to expose):
    o_t = sigmoid(W_io @ x_t + W_ho @ h_{t-1} + b_o)

Hidden state output:
    h_t = o_t * tanh(c_t)

Where:
- sigmoid(x) = 1/(1+exp(-x)) squashes to [0,1]
- * denotes element-wise (Hadamard) product
- tanh squashes to [-1,1]

Key Insight: The forget gate multiplicative interaction (f_t * c_{t-1}) enables
gradient flow without attenuation:
    dc_{t-1} = f_t * dc_{t}

If f_t ≈ 1, then gradients flow back with minimal degradation, solving the
vanishing gradient problem!

Why LSTM Works:
===============
1. Additive cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
   - Additive path allows gradients to flow without multiplicative decay
   - dL/dc_{t-1} = dL/dc_t * f_t (multiplication by ~1 instead of small values)

2. Three gates control information flow:
   - Input gate: learn what to store
   - Forget gate: learn what to discard
   - Output gate: learn what to expose

3. Gradient flow: Unlike Vanilla RNN where dh/dh_{t-1} involves W_hh^T ⊗ diag(...),
   LSTM has a shortcut path through additive cell state update

Mathematical Analysis:
======================
Gradient through cell state (ignoring input gates for simplicity):
    ∂c_t/∂c_{t-1} = f_t  (per-element multiplication)

Product over T steps:
    ∂c_T/∂c_0 = ∏_{t=1}^T f_t

For typical forget gate values around 0.5:
    ∏_{t=1}^T f_t ≈ 0.5^T (still decays exponentially!)

BUT: Forget gate can learn to be ~1.0 for important long-range dependencies
    If f_t ≈ 0.99: ∏_{t=1}^T ≈ 0.99^T (much better!)

This is why initializing forget gate bias to positive values (+1 or +2) is crucial:
- Biases forget gate to be "open" initially
- Network learns when to close (forget) information

Key Papers:
===========
1. "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
   - Original LSTM paper introducing gating mechanisms
   - Proves LSTM overcomes vanishing gradient problem
   - Shows experimental results on toy problems

2. "Learning Phrase Representations using RNN Encoder-Decoder for SMT"
   (Cho et al., 2014)
   - GRU paper (simpler alternative to LSTM)
   - Practical sequence-to-sequence results

3. "Generating Sequences With Recurrent Neural Networks"
   (Graves, 2013)
   - Practical LSTM implementation details
   - Mixture of experts for RNN outputs
   - Dropout in RNNs

4. "Recurrent Dropout without Memory Loss" (Gal & Ghahramani, 2016)
   - Correct way to apply dropout in RNNs
   - Variational dropout: same dropout mask across time

5. "On the State of the Art of Evaluation in Neural Language Models"
   (Melis et al., 2018)
   - Comprehensive LSTM tuning guide
   - Importance of initialization and regularization

6. "Densely Connected LSTM" (Huang et al., 2017)
   - Skip connections in stacked LSTMs
   - Improved information flow in deep models

Architecture Details:
====================

1. LSTM Cell (single timestep):
   - Inputs: x_t (batch, input_dim), h_{t-1} (batch, hidden_dim), c_{t-1} (batch, hidden_dim)
   - Parameters: 4 weight matrices (for 4 gates) + 4 bias vectors
   - Total parameters: (input_dim + hidden_dim + 1) * 4 * hidden_dim
   - Example: (100 + 128 + 1) * 4 * 128 = 66,816 parameters per cell

2. Multi-layer LSTM:
   - Stack LSTMs: output of one layer → input to next layer
   - Each layer maintains independent cell and hidden states
   - Enables learning hierarchical representations

3. Bidirectional LSTM:
   - Forward LSTM: processes sequence left-to-right
   - Backward LSTM: processes sequence right-to-left
   - Concatenate outputs: (batch, seq_len, 2*hidden_dim)

4. Attention variants:
   - Attention over cell states
   - Peephole connections: let gates look at cell state
   - Coupled input/forget gates (simplification)

Implementation Strategy:
=======================

Essential components:
- LSTMCell: single timestep with 4 gates
  * Pre-computation: combine x_t and h_{t-1} once, then slice for 4 gates
  * Numerically stable sigmoid/tanh implementations
  * Batch normalization option

- LSTM layer: sequence processing
  * Efficient: can use matrix operations for all timesteps at once
  * Variable sequence length handling with masks
  * Per-unit dropout/variational dropout

- Initialization:
  * Forget gate bias: uniform(1, 3) → forget gate starts open
  * Other biases: 0
  * Weights: Orthogonal or Xavier initialization
  * LSTM paper recommends: [-sqrt(1/hidden_dim), sqrt(1/hidden_dim)]

Common implementation tricks:
- Coupled input/forget gates: i_t + f_t = 1 (fewer parameters)
- Peephole connections: gates also look at c_t
- Layer normalization: normalize across feature dimension
- Gradient clipping: critical even with LSTM
- Zoneout: stochastically keep previous hidden/cell states

Performance:
============
- Parameters: ~4x Vanilla RNN (4 gates)
- Computation: ~4x Vanilla RNN per timestep
- Memory: same asymptotic complexity for storing states
- Forward pass still sequential (cannot parallelize across time)
- Bidirectional LSTM: requires two sequential passes

Variants:
=========
1. Peephole LSTM: gates see cell state c_t
   - 3 additional weight vectors per cell
   - Slightly improved performance in some tasks

2. Coupled Input-Forget (CIFG) LSTM:
   - i_t = 1 - f_t (input gate is inverse of forget gate)
   - Reduces parameters by ~25%
   - Often minimal performance loss

3. Layer Normalization LSTM:
   - Normalize gates before/after activation
   - Improved training stability
   - Allows higher learning rates

4. Gated Recurrent Unit (GRU):
   - Simpler LSTM with 2 gates instead of 3
   - ~75% parameters of LSTM
   - Similar performance in practice

Comparison with Alternatives:
=============================
vs Vanilla RNN:
  - LSTM solves vanishing gradient with 4x parameters
  - LSTM enables much longer effective context (~100+ vs ~5-10)
  - Both have sequential forward pass

vs GRU:
  - GRU simpler (2 gates vs 3) but often similar performance
  - GRU fewer parameters (75% of LSTM)
  - Choose GRU for efficiency, LSTM for maximum expressiveness

vs Transformer:
  - Transformer parallelizes across time with self-attention
  - LSTM fully sequential
  - Transformer requires more memory upfront
  - LSTM better for streaming/online processing
"""

from typing import Optional, Tuple
import numpy as np


class LSTMCell:
    """
    Single LSTM cell implementing one timestep of computation.

    Implements the 4-gate LSTM architecture:
    - Input gate (i_t): controls new information flow
    - Forget gate (f_t): controls information retention
    - Cell gate (g_t): candidate new information
    - Output gate (o_t): controls hidden state exposure
    """

    def __init__(self, input_dim: int, hidden_dim: int, forget_bias_init: float = 1.0):
        """
        Initialize LSTM cell.

        Args:
            input_dim: dimensionality of input
            hidden_dim: dimensionality of hidden state and cell state
            forget_bias_init: initial forget gate bias (important!)
                             Set to 1.0-2.0 to keep forget gate open initially
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_bias_init = forget_bias_init

        # Combined weight matrix for all 4 gates: 4 * hidden_dim output
        # W = [W_i; W_f; W_g; W_o] shape: (input_dim + hidden_dim, 4*hidden_dim)
        # TODO: Initialize with Xavier/Orthogonal
        self.W = None  # (input_dim + hidden_dim, 4*hidden_dim)
        self.b = None  # (4*hidden_dim,) with b[hidden_dim:2*hidden_dim] = forget_bias_init

        # Gradients
        self.dW = None
        self.db = None

    def forward(self, x_t: np.ndarray, h_t: np.ndarray, c_t: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Forward pass for single LSTM timestep.

        Args:
            x_t: input at time t, shape (batch_size, input_dim)
            h_t: hidden state at t-1, shape (batch_size, hidden_dim)
            c_t: cell state at t-1, shape (batch_size, hidden_dim)

        Returns:
            h_next: new hidden state, shape (batch_size, hidden_dim)
            c_next: new cell state, shape (batch_size, hidden_dim)
            cache: dict with values for backward pass
        """
        batch_size = x_t.shape[0]

        # TODO: Concatenate x_t and h_t for single matrix multiplication
        # x_h = np.concatenate([x_t, h_t], axis=1)  # (batch, input_dim + hidden_dim)

        # TODO: Compute all gate pre-activations in one matrix multiplication
        # gates = x_h @ self.W + self.b  # (batch, 4*hidden_dim)

        # TODO: Split into individual gates and apply activations
        # i_t = sigmoid(gates[:, 0:hidden_dim])
        # f_t = sigmoid(gates[:, hidden_dim:2*hidden_dim])
        # g_t = tanh(gates[:, 2*hidden_dim:3*hidden_dim])
        # o_t = sigmoid(gates[:, 3*hidden_dim:4*hidden_dim])

        # TODO: Cell state update: c_t = f_t * c_{t-1} + i_t * g_t
        # c_next = f_t * c_t + i_t * g_t

        # TODO: Hidden state: h_t = o_t * tanh(c_t)
        # h_next = o_t * np.tanh(c_next)

        # TODO: Store cache for backward pass
        # cache = {
        #     'x_t': x_t,
        #     'h_t': h_t,
        #     'c_t': c_t,
        #     'h_next': h_next,
        #     'c_next': c_next,
        #     'i_t': i_t,
        #     'f_t': f_t,
        #     'g_t': g_t,
        #     'o_t': o_t,
        # }

        pass

    def backward(self, dh_next: np.ndarray, dc_next: np.ndarray, cache: dict) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass through LSTM cell (single timestep).

        Args:
            dh_next: gradient w.r.t. hidden state at t+1
            dc_next: gradient w.r.t. cell state at t+1
            cache: cache from forward pass

        Returns:
            dh: gradient w.r.t. hidden state at t
            dc: gradient w.r.t. cell state at t
            dx: gradient w.r.t. input at t
        """
        # TODO: Extract cached values
        # TODO: Compute gradients through output gate
        # do = dh_next * tanh(c_next)

        # TODO: Compute gradients through cell state
        # Cell state gradient has two paths:
        # 1. From dc_next (from t+1)
        # 2. Through output gate from dh_next
        # dc = dc_next + dh_next * o_t * (1 - tanh(c_next)**2)

        # TODO: Compute gradients through input/forget/cell gates
        # di = dc * g_t
        # df = dc * c_t
        # dg = dc * i_t

        # TODO: Compute gradients w.r.t. pre-activations (with sigmoid/tanh derivatives)
        # di_pre = di * i_t * (1 - i_t)
        # df_pre = df * f_t * (1 - f_t)
        # dg_pre = dg * (1 - g_t**2)
        # do_pre = do * o_t * (1 - o_t)

        # TODO: Compute gradient w.r.t. weights and inputs
        # TODO: Concatenate gate gradients: [di_pre, df_pre, dg_pre, do_pre]
        # TODO: Return dx, dh_t, dc_t for next timestep

        pass


class LSTM:
    """
    Full LSTM layer processing entire sequences.

    Handles multiple timesteps with state management.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1, bidirectional: bool = False,
                 dropout: float = 0.0):
        """
        Initialize LSTM layer.

        Args:
            input_dim: dimensionality of input sequences
            hidden_dim: dimensionality of hidden/cell states
            output_dim: dimensionality of output
            num_layers: number of stacked LSTM layers
            bidirectional: whether to use bidirectional LSTM
            dropout: dropout rate (applied between layers, not within)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # TODO: Initialize LSTM cells for each layer
        # For num_layers > 1:
        #   Layer 0: input_dim -> hidden_dim
        #   Layer i>0: hidden_dim -> hidden_dim
        self.cells = []

        # TODO: Initialize output projection
        self.W_hy = None  # (hidden_dim, output_dim)
        self.b_y = None   # (output_dim,)

    def forward(self, X: np.ndarray, h0: Optional[np.ndarray] = None,
                c0: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through entire sequence.

        Args:
            X: input sequences, shape (batch_size, seq_len, input_dim)
            h0: initial hidden state, shape (batch_size, hidden_dim)
            c0: initial cell state, shape (batch_size, hidden_dim)

        Returns:
            outputs: all hidden states, shape (batch_size, seq_len, hidden_dim)
            h_final: final hidden state, shape (batch_size, hidden_dim)
            c_final: final cell state, shape (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = X.shape

        if h0 is None:
            h0 = np.zeros((batch_size, self.hidden_dim))
        if c0 is None:
            c0 = np.zeros((batch_size, self.hidden_dim))

        # TODO: Initialize outputs and state arrays
        # TODO: For each timestep:
        #   1. Pass through all layers sequentially
        #   2. Maintain hidden and cell states for each layer
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
        # TODO: Initialize hidden and cell state gradients
        # TODO: Backpropagate through output projection
        # TODO: For each timestep (in reverse):
        #   1. Backpropagate through all layers
        #   2. Accumulate weight gradients across timesteps
        # TODO: Apply gradient clipping (important!)
        # TODO: Update weights

        pass


class BidirectionalLSTM(LSTM):
    """
    Bidirectional LSTM processing sequences in both directions.

    Useful for tasks where full sequence context is available
    (classification, tagging, not generation).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1, dropout: float = 0.0):
        """
        Initialize Bidirectional LSTM.

        Args:
            input_dim: dimensionality of input sequences
            hidden_dim: dimensionality of each direction's hidden state
            output_dim: dimensionality of output
            num_layers: number of stacked layers
            dropout: dropout rate between layers
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers,
                         bidirectional=True, dropout=dropout)

        # TODO: Create separate forward and backward LSTM cells
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


class PeepholeLSTM(LSTM):
    """
    LSTM with peephole connections (Jozefowicz et al., 2015).

    Gates can see the cell state directly, improving fine-grained control.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1):
        """
        Initialize Peephole LSTM.

        Args:
            input_dim: dimensionality of input sequences
            hidden_dim: dimensionality of hidden/cell states
            output_dim: dimensionality of output
            num_layers: number of stacked layers
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers)

        # TODO: Add peephole weight vectors for input, forget, output gates
        # Each is (hidden_dim,) to scale cell state before gate computation
        self.w_ci = None  # Peephole weight for input gate
        self.w_cf = None  # Peephole weight for forget gate
        self.w_co = None  # Peephole weight for output gate

    def forward(self, X: np.ndarray, h0: Optional[np.ndarray] = None,
                c0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass with peephole connections.

        Gates now look at cell state:
            i_t = sigmoid(W_i @ x_t + W_h @ h_{t-1} + w_ci ⊙ c_{t-1} + b_i)
            f_t = sigmoid(W_f @ x_t + W_h @ h_{t-1} + w_cf ⊙ c_{t-1} + b_f)
            o_t = sigmoid(W_o @ x_t + W_h @ h_{t-1} + w_co ⊙ c_t + b_o)

        Args:
            X: input sequences
            h0: initial hidden state
            c0: initial cell state

        Returns:
            outputs: all hidden states
            h_final: final hidden state
            c_final: final cell state
        """
        # TODO: Similar to LSTM.forward but with peephole modifications
        # TODO: Gates include additional term: w_c ⊙ c_t (element-wise multiply)

        pass


if __name__ == "__main__":
    # Test basic LSTM
    batch_size, seq_len, input_dim, hidden_dim, output_dim = 32, 10, 50, 100, 10

    # TODO: Create sample data
    # X = np.random.randn(batch_size, seq_len, input_dim)

    # TODO: Create LSTM model
    # lstm = LSTM(input_dim, hidden_dim, output_dim)
    # outputs, h_final, c_final = lstm.forward(X)
    # print(f"Output shape: {outputs.shape}")
    # print(f"Final hidden shape: {h_final.shape}")
    # print(f"Final cell shape: {c_final.shape}")

    # TODO: Test bidirectional LSTM
    # bilstm = BidirectionalLSTM(input_dim, hidden_dim, output_dim)
    # outputs = bilstm.forward(X)
    # print(f"Bidirectional output shape: {outputs.shape}")
