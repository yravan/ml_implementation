"""
Vanilla Recurrent Neural Network (RNN) Implementation

Comprehensive implementation of a basic RNN sequence model with full hidden state
management and bidirectional support.

Theory:
========
The Vanilla RNN is the foundation of all recurrent architectures. At each timestep t,
the RNN maintains a hidden state h_t that captures information about all previous
inputs. The recurrence relation is:

    h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
    y_t = W_hy @ h_t + b_y

Where:
- h_t: hidden state at time t (shape: [batch_size, hidden_dim])
- x_t: input at time t (shape: [batch_size, input_dim])
- W_hh: hidden-to-hidden weight matrix (shape: [hidden_dim, hidden_dim])
- W_xh: input-to-hidden weight matrix (shape: [input_dim, hidden_dim])
- W_hy: hidden-to-output weight matrix (shape: [hidden_dim, output_dim])
- tanh: nonlinearity that squashes values to [-1, 1]

For a sequence of length T, we iterate this recurrence T times:
- Process x_0 through x_{T-1}
- Accumulate hidden states h_1 through h_T
- Output either the final hidden state (for classification) or all hidden states
  (for sequence tagging/generation)

Limitations:
- Vanishing/exploding gradient problem: gradients decay exponentially with depth
  due to repeated multiplication of W_hh during backpropagation
- Cannot maintain long-term dependencies effectively (effective range ~5-10 steps)
- tanh saturation leads to dead neurons

Mathematical Formulation:
========================
Forward Pass:
    h_0 = 0 (or learned initial state)
    For t = 0 to T-1:
        a_t = W_xh @ x_t + W_hh @ h_t + b_h
        h_{t+1} = tanh(a_t)
    y_t = W_hy @ h_{t+1} + b_y

Loss (cross-entropy for classification):
    L = -sum_{t} log(softmax(y_t)[y_t^true])

Backpropagation Through Time (BPTT):
    dL/dW_hh requires summing gradients across all timesteps
    dh_{t}/dh_{t-1} = W_hh^T @ diag(1 - h_t^2)  (chain rule with tanh derivative)

Gradient flow: dL/dh_0 requires multiplying (T-1) Jacobian matrices, causing
    ||dL/dh_0|| ~ ||product_{t=1}^{T-1} (W_hh^T @ diag(...))|| which vanishes or explodes

Key Papers:
===========
1. "A Critical Review of Recurrent Neural Networks for Sequence Learning"
   (Lipton et al., 2015)
   - Comprehensive analysis of RNN capabilities and limitations
   - Discussion of vanishing/exploding gradient problem
   - Empirical comparisons

2. "On the difficulty of training Recurrent Neural Networks"
   (Pascanu et al., 2013)
   - Mathematical analysis of gradient flow
   - Introduction of gradient clipping as solution
   - Analysis of saturation regions

3. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical MT"
   (Cho et al., 2014)
   - Practical RNN implementation for sequence-to-sequence tasks
   - Early work on encoder-decoder architectures

4. "Sequence to Sequence Learning with Neural Networks"
   (Sutskever et al., 2014)
   - Large-scale RNN application
   - Reverse input sequences trick
   - Introduces LSTM for better gradient flow

Architecture Details:
====================

1. RNN Cell (single timestep):
   - Input shape: (batch_size, input_dim)
   - Hidden state shape: (batch_size, hidden_dim)
   - Weights: W_xh, W_hh, W_hy with appropriate dimensions
   - Uses tanh activation (common alternative: ReLU for GRU-style)

2. RNN Layer (full sequence):
   - Processes entire sequence in loop
   - Returns all hidden states or just final state
   - Supports bidirectional processing (forward + backward RNN)
   - Can be stacked for deeper models

3. Bidirectional RNN:
   - Forward RNN: h_f_t = RNN_f(x_0, ..., x_t)
   - Backward RNN: h_b_t = RNN_b(x_T, ..., x_t)
   - Concatenate: h_t = [h_f_t; h_b_t]
   - Better context but no parallelization across time

Implementation Strategy:
=======================

Essential components:
- RNNCell class: single timestep computation
  * Parameters: weight initialization (Xavier/He)
  * Forward: h_{t+1} = tanh(W_hh @ h_t + W_xh @ x_t + b)
  * Numerically stable implementation

- RNN layer class: wraps RNNCell for sequences
  * Handle variable sequence lengths with padding masks
  * Return all outputs or just final state
  * Support bidirectional mode

- Initialization:
  * W_hh ~ Uniform(-1/sqrt(H), 1/sqrt(H)) where H = hidden_dim
  * W_xh ~ Uniform(-sqrt(6)/(sqrt(input_dim + hidden_dim)), ...)
  * Biases: initialized to 0

Common implementation issues:
- Gradient clipping required (clip by global norm or value)
- Careful handling of variable sequence lengths
- Memory efficient backpropagation (truncated BPTT)
- Proper masking for padded sequences

Usage Example:
==============
    # Single layer RNN for classification
    rnn = VanillaRNN(input_dim=50, hidden_dim=100, output_dim=10)
    hidden_states = rnn.forward(sequences)  # (batch, seq_len, hidden_dim)
    logits = rnn.classify(hidden_states[:, -1, :])  # use final state

    # Bidirectional RNN for sequence tagging
    birnn = BidirectionalRNN(input_dim=50, hidden_dim=100, output_dim=20)
    outputs = birnn.forward(sequences)  # (batch, seq_len, 2*hidden_dim)

    # Stacked RNN
    stacked = nn.Sequential(
        VanillaRNN(50, 100),
        VanillaRNN(100, 100),
        VanillaRNN(100, 10)
    )

Performance Characteristics:
============================
- Time complexity: O(T * H^2) where T = sequence_length, H = hidden_dim
- Space complexity: O(T * B * H) for storing all hidden states
- Gradient computation: O(T * H^2) due to BPTT
- Forward pass fully sequential: cannot parallelize across time
- Bidirectional requires two sequential passes

Comparison with Alternatives:
=============================
vs LSTM: LSTM has gating mechanisms that enable better long-range dependencies
        but with ~4x parameters and slightly higher computation
vs GRU:  Similar to LSTM but with fewer parameters (~3x vs 1x Vanilla)
vs Transformer: Transformer parallelizes across time with self-attention but
               requires more memory and computation upfront
"""

from typing import Optional, Tuple
import numpy as np


class RNNCell:
    """
    Single RNN cell for processing one timestep.

    Implements: h_{t+1} = tanh(W_hh @ h_t + W_xh @ x_t + b_h)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize RNN cell.

        Args:
            input_dim: dimensionality of input x_t
            hidden_dim: dimensionality of hidden state h_t
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Weight matrices
        # TODO: Implement Xavier initialization
        self.W_xh = None  # (input_dim, hidden_dim)
        self.W_hh = None  # (hidden_dim, hidden_dim)
        self.b_h = None   # (hidden_dim,)

        # Gradients
        self.dW_xh = None
        self.dW_hh = None
        self.db_h = None

    def forward(self, x_t: np.ndarray, h_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for single timestep.

        Args:
            x_t: input at time t, shape (batch_size, input_dim)
            h_t: hidden state at time t-1, shape (batch_size, hidden_dim)

        Returns:
            h_{t+1}: new hidden state, shape (batch_size, hidden_dim)
            cache: tuple of (x_t, h_t, h_{t+1}) for backprop
        """
        # TODO: Compute a_t = W_xh @ x_t + W_hh @ h_t + b_h
        # TODO: Compute h_{t+1} = tanh(a_t)
        # TODO: Store cache for backward pass
        # TODO: Return h_{t+1}, cache
        pass

    def backward(self, dh_next: np.ndarray, cache: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for single timestep.

        Args:
            dh_next: gradient w.r.t. hidden state at time t+1
            cache: output from forward pass

        Returns:
            dh: gradient w.r.t. hidden state at time t
            dx: gradient w.r.t. input at time t
            dW_xh, dW_hh, db_h: gradients w.r.t. parameters
        """
        # TODO: Implement chain rule with tanh derivative
        # TODO: d(tanh)/da = 1 - tanh^2(a)
        # TODO: Accumulate weight gradients
        pass


class VanillaRNN:
    """
    Multi-layer Vanilla RNN for processing sequences.

    Stacks multiple RNN cells across time and potentially across layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1, bidirectional: bool = False):
        """
        Initialize Vanilla RNN.

        Args:
            input_dim: dimensionality of input sequences
            hidden_dim: dimensionality of hidden states
            output_dim: dimensionality of output
            num_layers: number of stacked RNN layers
            bidirectional: whether to use bidirectional RNN
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Initialize RNN cells for each layer
        # TODO: Create num_layers RNNCell instances
        self.cells = []

        # Output projection
        # TODO: Initialize W_hy and b_y
        self.W_hy = None
        self.b_y = None

    def forward(self, X: np.ndarray, h0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through entire sequence.

        Args:
            X: input sequences, shape (batch_size, seq_len, input_dim)
            h0: initial hidden state, shape (batch_size, hidden_dim)

        Returns:
            outputs: hidden states at each timestep (batch_size, seq_len, hidden_dim)
            h_final: final hidden state (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = X.shape

        if h0 is None:
            h0 = np.zeros((batch_size, self.hidden_dim))

        # TODO: Implement forward pass
        # 1. Process each timestep with RNNCell
        # 2. Maintain hidden state across timesteps
        # 3. Store all outputs
        # 4. Return outputs and final hidden state
        # 5. Store cache for backward pass (if needed)
        pass

    def backward(self, doutputs: np.ndarray, learn_rate: float = 0.01) -> None:
        """
        Backward pass through entire sequence (BPTT).

        Args:
            doutputs: gradient w.r.t. outputs from all timesteps
            learn_rate: learning rate for weight updates
        """
        # TODO: Implement BPTT
        # 1. Initialize gradient for final hidden state
        # 2. Loop backward through timesteps
        # 3. Accumulate gradients for shared weights
        # 4. Apply gradient clipping to prevent explosion
        # 5. Update weights
        pass

    def classify(self, sequences: np.ndarray) -> np.ndarray:
        """
        Classification using final hidden state.

        Args:
            sequences: input sequences (batch_size, seq_len, input_dim)

        Returns:
            logits: class logits (batch_size, output_dim)
        """
        # TODO: Forward pass through RNN
        # TODO: Extract final hidden state
        # TODO: Project through W_hy + b_y
        pass


class BidirectionalRNN(VanillaRNN):
    """
    Bidirectional Vanilla RNN that processes sequence forward and backward.

    Concatenates forward and backward hidden states at each position.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1):
        """
        Initialize Bidirectional RNN.

        Args:
            input_dim: dimensionality of input sequences
            hidden_dim: dimensionality of each RNN's hidden states
            output_dim: dimensionality of output
            num_layers: number of stacked bidirectional RNN layers
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, bidirectional=True)

        # TODO: Create separate forward and backward RNN cells
        self.forward_cells = []
        self.backward_cells = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through sequence in both directions.

        Args:
            X: input sequences, shape (batch_size, seq_len, input_dim)

        Returns:
            outputs: concatenated hidden states (batch_size, seq_len, 2*hidden_dim)
        """
        # TODO: Process forward direction: x_0 -> x_1 -> ... -> x_T
        # TODO: Process backward direction: x_T -> x_{T-1} -> ... -> x_0
        # TODO: Concatenate forward and backward hidden states at each position
        # TODO: Return combined outputs
        pass


class StackedRNN(VanillaRNN):
    """
    Stacked Vanilla RNN with multiple layers.

    Output of one layer becomes input to next layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, bidirectional: bool = False):
        """
        Initialize Stacked RNN.

        Args:
            input_dim: dimensionality of input sequences
            hidden_dim: dimensionality of hidden states in each layer
            output_dim: dimensionality of output
            num_layers: number of stacked layers
            bidirectional: whether each layer is bidirectional
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, bidirectional)

        # TODO: Initialize num_layers RNN layers
        # TODO: Layer i takes input of shape (..., hidden_dim) from layer i-1
        # TODO: (except layer 0 which takes input_dim input)
        self.layers = []

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through all layers.

        Args:
            X: input sequences, shape (batch_size, seq_len, input_dim)

        Returns:
            outputs: final layer outputs (batch_size, seq_len, hidden_dim)
            hidden_states: list of hidden states from each layer
        """
        # TODO: Process through each layer sequentially
        # TODO: Pass output of layer i to layer i+1
        # TODO: Return final outputs and all hidden states
        pass


if __name__ == "__main__":
    # Test basic RNN
    batch_size, seq_len, input_dim, hidden_dim, output_dim = 32, 10, 50, 100, 10

    # TODO: Create sample data
    X = np.random.randn(batch_size, seq_len, input_dim)

    # TODO: Create RNN model
    # rnn = VanillaRNN(input_dim, hidden_dim, output_dim)
    # outputs = rnn.forward(X)
    # print(f"Output shape: {outputs[0].shape}")

    # TODO: Test bidirectional RNN
    # birnn = BidirectionalRNN(input_dim, hidden_dim, output_dim)
    # outputs = birnn.forward(X)
    # print(f"Bidirectional output shape: {outputs.shape}")
