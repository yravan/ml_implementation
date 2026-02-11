"""
Recurrent Neural Network Modules and Functional Operations

This module consolidates all recurrent operations:
- Module classes: RNNCell, LSTMCell, GRUCell, BidirectionalRNNCell, etc.
- Functional classes: RNNCellFunction, LSTMCellFunction, GRUCellFunction, BidirectionalFunction
- Helper functions: sigmoid, tanh

References:
    - RNN: "A critical review of recurrent neural networks for sequence learning"
      (Lipton et al., 2015) https://arxiv.org/abs/1506.00019
    - LSTM: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
      https://direct.mit.edu/neco/article/9/8/1735/6109
    - GRU: "Learning Phrase Representations using RNN Encoder-Decoder"
      (Cho et al., 2014) https://arxiv.org/abs/1406.1078
"""

import numpy as np
import math
from typing import Tuple, Optional, Union

from .module import Module, Parameter
from ..foundations.functionals import Function
from ..foundations import Tensor


# ============================================================================
# Helper Functions
# ============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent."""
    return np.tanh(x)


# ============================================================================
# MODULE CLASSES (Non-Trainable Parameter Containers)
# ============================================================================

class RNNCell(Module):
    """
    Basic (Vanilla) RNN Cell.

    Implements a single step of recurrent neural network:
        h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_h)
        y_t = W_ho @ h_t + b_o

    Attributes:
        d_in (int): Input dimension
        d_h (int): Hidden state dimension
        d_out (int): Output dimension
        W_ih (Parameter): Input-to-hidden weight [d_h, d_in]
        W_hh (Parameter): Hidden-to-hidden weight (recurrent) [d_h, d_h]
        b_h (Parameter): Hidden bias [d_h]
        W_ho (Parameter): Hidden-to-output weight [d_out, d_h]
        b_o (Parameter): Output bias [d_out]
    """

    def __init__(self, d_in, d_h, d_out):
        """
        Initialize RNN cell.

        Args:
            d_in (int): Input dimension
            d_h (int): Hidden dimension
            d_out (int): Output dimension
        """
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out

        # Weight initialization (Xavier/Glorot)
        # For RNN, spectral radius of W_hh should be ~1 for stability
        init_scale_ih = math.sqrt(2.0 / (d_in + d_h))
        init_scale_hh = math.sqrt(2.0 / (2 * d_h))  # More conservative for recurrent
        init_scale_ho = math.sqrt(2.0 / (d_h + d_out))

        self.W_ih = Parameter(np.random.randn(d_h, d_in) * init_scale_ih)
        self.W_hh = Parameter(np.random.randn(d_h, d_h) * init_scale_hh)
        self.b_h = Parameter(np.zeros(d_h))

        self.W_ho = Parameter(np.random.randn(d_out, d_h) * init_scale_ho)
        self.b_o = Parameter(np.zeros(d_out))

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tuple[Tensor, Tensor, dict]:
        """
        Forward pass for single time step.

        Args:
            x_t (Tensor): Input at time t [batch_size, d_in]
            h_prev (Tensor): Hidden state from previous time step [batch_size, d_h]

        Returns:
            y_t (Tensor): Output at time t [batch_size, d_out]
            h_t (Tensor): Hidden state at time t [batch_size, d_h]
            cache (dict): Cache for backward pass

        Algorithm:
            1. Compute hidden pre-activation:
               z_h = x_t @ W_ih^T + h_prev @ W_hh^T + b_h
               Shape: [batch_size, d_h]

            2. Apply tanh activation:
               h_t = tanh(z_h)
               Shape: [batch_size, d_h]

            3. Compute output:
               y_t = h_t @ W_ho^T + b_o
               Shape: [batch_size, d_out]

            4. Cache values for backward pass
        """
        raise NotImplementedError(
            "RNNCell.forward() requires implementation.\n"
            "Hints:\n"
            "  1. Convert Tensor inputs to numpy if needed (use .data attribute)\n"
            "  2. Compute hidden pre-activation:\n"
            "     z_h = x_t.data @ self.W_ih.data.T + h_prev.data @ self.W_hh.data.T + self.b_h.data\n"
            "     Shape: [batch_size, d_h]\n"
            "  3. Apply tanh:\n"
            "     h_t = np.tanh(z_h)\n"
            "  4. Compute output:\n"
            "     y_t = h_t @ self.W_ho.data.T + self.b_o.data\n"
            "     Shape: [batch_size, d_out]\n"
            "  5. Wrap outputs as Tensors if needed\n"
            "  6. Cache for backward:\n"
            "     cache = {'x_t': x_t, 'h_prev': h_prev, 'h_t': h_t,\n"
            "              'z_h': z_h, 'tanh_deriv': 1 - h_t**2}\n"
            "  7. Return y_t, h_t, cache\n"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"d_in={self.d_in}, d_h={self.d_h}, d_out={self.d_out}"


class BidirectionalRNNCell(Module):
    """
    Bidirectional RNN: processes sequence in both forward and backward directions.

    Useful for tasks where context from both past and future is important
    (e.g., BERT, BiLSTM tagging).

    EQUATIONS:
        Forward pass: h_f_t = RNN_f(x_t, h_f_{t-1})
        Backward pass: h_b_t = RNN_b(x_t, h_b_{t+1})
        Output: y_t = [h_f_t; h_b_t]  (concatenated)

    Output dimension: 2 * d_h (concatenation of forward and backward)
    """

    def __init__(self, d_in, d_h, d_out):
        """
        Initialize bidirectional RNN.

        Args:
            d_in (int): Input dimension
            d_h (int): Hidden dimension per direction
            d_out (int): Output dimension
        """
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out

        # Forward and backward RNN cells
        self.rnn_forward = RNNCell(d_in, d_h, d_h)
        self.rnn_backward = RNNCell(d_in, d_h, d_h)

        # Output projection for concatenated [h_f; h_b]
        init_scale = math.sqrt(2.0 / (2 * d_h + d_out))
        self.W_out = Parameter(np.random.randn(d_out, 2 * d_h) * init_scale)
        self.b_out = Parameter(np.zeros(d_out))

    def forward(self, sequence: Tensor) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """
        Forward pass through bidirectional RNN.

        Args:
            sequence (Tensor): Input sequence [T, batch_size, d_in]
                              or [batch_size, T, d_in]

        Returns:
            output (Tensor): Bidirectional output [T, batch_size, d_out]
            h_forward (Tensor): Forward hidden states [T, batch_size, d_h]
            h_backward (Tensor): Backward hidden states [T, batch_size, d_h]
            cache (dict): Cache for backward pass
        """
        raise NotImplementedError(
            "BidirectionalRNNCell.forward() requires implementation.\n"
            "Hints:\n"
            "  1. Reshape input if needed: ensure [T, batch_size, d_in]\n"
            "  2. Forward pass (t=0 to T-1):\n"
            "     h_f = zeros([batch_size, d_h])\n"
            "     h_forward = []\n"
            "     for t in range(T):\n"
            "       y_f, h_f, _ = rnn_forward.forward(sequence[t], h_f)\n"
            "       h_forward.append(h_f)\n"
            "  3. Backward pass (t=T-1 to 0):\n"
            "     h_b = zeros([batch_size, d_h])\n"
            "     h_backward = [None] * T\n"
            "     for t in range(T-1, -1, -1):\n"
            "       y_b, h_b, _ = rnn_backward.forward(sequence[t], h_b)\n"
            "       h_backward[t] = h_b\n"
            "  4. Stack and concatenate results\n"
            "  5. Return as Tensors\n"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"d_in={self.d_in}, d_h={self.d_h}, d_out={self.d_out}"


class LSTMCell(Module):
    """
    Long Short-Term Memory (LSTM) Cell.

    Implements single time step of LSTM with gates controlling information flow.

    Attributes:
        d_in (int): Input dimension
        d_h (int): Hidden/cell state dimension
        d_out (int): Output dimension

        Weight matrices (d_h × [d_in + d_h] for each gate + cell):
        - Forget gate: W_if [d_h, d_in], W_hf [d_h, d_h], b_f [d_h]
        - Input gate: W_ii [d_h, d_in], W_hi [d_h, d_h], b_i [d_h]
        - Output gate: W_io [d_h, d_in], W_ho [d_h, d_h], b_o [d_h]
        - Cell candidate: W_ic [d_h, d_in], W_hc [d_h, d_h], b_c [d_h]
        - Output projection: W_hy [d_out, d_h], b_y [d_out]
    """

    def __init__(self, d_in, d_h, d_out):
        """
        Initialize LSTM cell.

        Args:
            d_in (int): Input dimension
            d_h (int): Hidden dimension (cell state dimension)
            d_out (int): Output dimension
        """
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out

        # Weight initialization scales
        init_scale_input = math.sqrt(2.0 / (d_in + d_h))
        init_scale_hidden = math.sqrt(2.0 / (2 * d_h))  # Orthogonal-like
        init_scale_output = math.sqrt(2.0 / (d_h + d_out))

        # Forget gate: controls what to forget from previous cell state
        self.W_if = Parameter(np.random.randn(d_h, d_in) * init_scale_input)
        self.W_hf = Parameter(np.random.randn(d_h, d_h) * init_scale_hidden)
        self.b_f = Parameter(np.ones(d_h) * 0.0)  # Often initialize forget bias to 1.0 or 0.1

        # Input gate: controls what new input to accept
        self.W_ii = Parameter(np.random.randn(d_h, d_in) * init_scale_input)
        self.W_hi = Parameter(np.random.randn(d_h, d_h) * init_scale_hidden)
        self.b_i = Parameter(np.zeros(d_h))

        # Output gate: controls what to expose as hidden state
        self.W_io = Parameter(np.random.randn(d_h, d_in) * init_scale_input)
        self.W_ho = Parameter(np.random.randn(d_h, d_h) * init_scale_hidden)
        self.b_o = Parameter(np.zeros(d_h))

        # Candidate cell state: proposed new information
        self.W_ic = Parameter(np.random.randn(d_h, d_in) * init_scale_input)
        self.W_hc = Parameter(np.random.randn(d_h, d_h) * init_scale_hidden)
        self.b_c = Parameter(np.zeros(d_h))

        # Output projection
        self.W_hy = Parameter(np.random.randn(d_out, d_h) * init_scale_output)
        self.b_y = Parameter(np.zeros(d_out))

    def forward(self, x_t: Tensor, h_prev: Tensor, c_prev: Tensor) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """
        Forward pass for single LSTM time step.

        Args:
            x_t (Tensor): Input at time t [batch_size, d_in]
            h_prev (Tensor): Hidden state from previous time step [batch_size, d_h]
            c_prev (Tensor): Cell state from previous time step [batch_size, d_h]

        Returns:
            y_t (Tensor): Output at time t [batch_size, d_out]
            h_t (Tensor): Hidden state at time t [batch_size, d_h]
            c_t (Tensor): Cell state at time t [batch_size, d_h]
            cache (dict): Cache for backward pass
        """
        raise NotImplementedError(
            "LSTMCell.forward() requires implementation.\n"
            "Hints:\n"
            "  1. Convert Tensor inputs to numpy if needed\n"
            "  2. Forget gate:\n"
            "     z_f = x_t.data @ self.W_if.data.T + h_prev.data @ self.W_hf.data.T + self.b_f.data\n"
            "     f_t = sigmoid(z_f)\n"
            "  3. Input gate:\n"
            "     z_i = x_t.data @ self.W_ii.data.T + h_prev.data @ self.W_hi.data.T + self.b_i.data\n"
            "     i_t = sigmoid(z_i)\n"
            "  4. Cell candidate:\n"
            "     z_c = x_t.data @ self.W_ic.data.T + h_prev.data @ self.W_hc.data.T + self.b_c.data\n"
            "     c_tilde = np.tanh(z_c)\n"
            "  5. Cell state update (CRITICAL):\n"
            "     c_t = f_t * c_prev.data + i_t * c_tilde  (element-wise multiply)\n"
            "  6. Output gate:\n"
            "     z_o = x_t.data @ self.W_io.data.T + h_prev.data @ self.W_ho.data.T + self.b_o.data\n"
            "     o_t = sigmoid(z_o)\n"
            "  7. Hidden state:\n"
            "     h_t = o_t * np.tanh(c_t)  (element-wise multiply)\n"
            "  8. Output projection:\n"
            "     y_t = h_t @ self.W_hy.data.T + self.b_y.data\n"
            "  9. Wrap as Tensors and cache\n"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"d_in={self.d_in}, d_h={self.d_h}, d_out={self.d_out}"


class StackedLSTM(Module):
    """
    Multiple LSTM layers stacked vertically.

    Each layer processes output from previous layer:
        Layer 1: input -> hidden_1
        Layer 2: hidden_1 -> hidden_2
        ...
        Layer n: hidden_{n-1} -> output
    """

    def __init__(self, d_in, d_h, num_layers, d_out):
        """
        Initialize stacked LSTM.

        Args:
            d_in (int): Input dimension
            d_h (int): Hidden dimension (same for all layers)
            num_layers (int): Number of LSTM layers
            d_out (int): Output dimension
        """
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.num_layers = num_layers
        self.d_out = d_out

        # Create LSTM cells for each layer
        self.lstm_cells = []
        for l in range(num_layers):
            input_dim = d_in if l == 0 else d_h
            output_dim = d_out if l == num_layers - 1 else d_h
            self.lstm_cells.append(LSTMCell(input_dim, d_h, output_dim))

    def forward(self, sequence: Tensor) -> Tuple[Tensor, list, list]:
        """
        Forward pass through stacked LSTM.

        Args:
            sequence (Tensor): Input sequence [T, batch_size, d_in]

        Returns:
            output (Tensor): Output sequence [T, batch_size, d_out]
            hidden_states (list): Hidden states per layer [num_layers][T, batch_size, d_h]
            cell_states (list): Cell states per layer [num_layers][T, batch_size, d_h]
        """
        raise NotImplementedError(
            "StackedLSTM.forward() requires implementation.\n"
            "Hints:\n"
            "  1. Initialize hidden/cell states for all layers\n"
            "  2. For each time step t:\n"
            "     For each layer l:\n"
            "       If l=0: input to layer is x_t (sequence[t])\n"
            "       Else: input to layer is h_{l-1}_t (output from layer l-1)\n"
            "       y_l_t, h_l_t, c_l_t = lstm_cells[l].forward(...)\n"
            "  3. Return final output and all hidden/cell states as Tensors\n"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"d_in={self.d_in}, d_h={self.d_h}, num_layers={self.num_layers}, d_out={self.d_out}"


class GRUCell(Module):
    """
    Gated Recurrent Unit (GRU) Cell.

    Simplified version of LSTM with 2 gates instead of 3, and hidden state
    serving as both the state and the output.

    Attributes:
        d_in (int): Input dimension
        d_h (int): Hidden state dimension
        d_out (int): Output dimension

        Weight matrices for reset and update gates, plus candidate:
        - Reset gate: W_ir [d_h, d_in], W_hr [d_h, d_h], b_r [d_h]
        - Update gate: W_iz [d_h, d_in], W_hz [d_h, d_h], b_z [d_h]
        - Candidate: W_ih [d_h, d_in], W_hh [d_h, d_h], b_h [d_h]
        - Output: W_hy [d_out, d_h], b_y [d_out]
    """

    def __init__(self, d_in, d_h, d_out):
        """
        Initialize GRU cell.

        Args:
            d_in (int): Input dimension
            d_h (int): Hidden dimension
            d_out (int): Output dimension
        """
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out

        # Weight initialization scales
        init_scale_input = math.sqrt(2.0 / (d_in + d_h))
        init_scale_hidden = math.sqrt(2.0 / (2 * d_h))
        init_scale_output = math.sqrt(2.0 / (d_h + d_out))

        # Reset gate: controls what to reset
        self.W_ir = Parameter(np.random.randn(d_h, d_in) * init_scale_input)
        self.W_hr = Parameter(np.random.randn(d_h, d_h) * init_scale_hidden)
        self.b_r = Parameter(np.zeros(d_h))

        # Update gate: controls what to update (interpolation)
        self.W_iz = Parameter(np.random.randn(d_h, d_in) * init_scale_input)
        self.W_hz = Parameter(np.random.randn(d_h, d_h) * init_scale_hidden)
        self.b_z = Parameter(np.zeros(d_h))

        # Candidate hidden state
        self.W_ih = Parameter(np.random.randn(d_h, d_in) * init_scale_input)
        self.W_hh = Parameter(np.random.randn(d_h, d_h) * init_scale_hidden)
        self.b_h = Parameter(np.zeros(d_h))

        # Output projection
        self.W_hy = Parameter(np.random.randn(d_out, d_h) * init_scale_output)
        self.b_y = Parameter(np.zeros(d_out))

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tuple[Tensor, Tensor, dict]:
        """
        Forward pass for single GRU time step.

        Args:
            x_t (Tensor): Input at time t [batch_size, d_in]
            h_prev (Tensor): Hidden state from previous time step [batch_size, d_h]

        Returns:
            y_t (Tensor): Output at time t [batch_size, d_out]
            h_t (Tensor): Hidden state at time t [batch_size, d_h]
            cache (dict): Cache for backward pass
        """
        raise NotImplementedError(
            "GRUCell.forward() requires implementation.\n"
            "Hints:\n"
            "  1. Convert Tensor inputs to numpy if needed\n"
            "  2. Reset gate:\n"
            "     z_r = x_t.data @ self.W_ir.data.T + h_prev.data @ self.W_hr.data.T + self.b_r.data\n"
            "     r_t = sigmoid(z_r)\n"
            "  3. Update gate:\n"
            "     z_z = x_t.data @ self.W_iz.data.T + h_prev.data @ self.W_hz.data.T + self.b_z.data\n"
            "     z_t = sigmoid(z_z)\n"
            "  4. Candidate (with reset applied):\n"
            "     z_h = x_t.data @ self.W_ih.data.T + (r_t * h_prev.data) @ self.W_hh.data.T + self.b_h.data\n"
            "     h_tilde = np.tanh(z_h)\n"
            "  5. Hidden state update (interpolation):\n"
            "     h_t = z_t * h_prev.data + (1 - z_t) * h_tilde\n"
            "  6. Output:\n"
            "     y_t = h_t @ self.W_hy.data.T + self.b_y.data\n"
            "  7. Wrap as Tensors and cache\n"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"d_in={self.d_in}, d_h={self.d_h}, d_out={self.d_out}"


# ============================================================================
# Sequence-Level RNN Wrappers (Process Full Sequences)
# ============================================================================

class RNN(Module):
    """
    Multi-layer RNN module that processes full sequences.

    Wraps RNNCell to process entire sequences and supports multiple layers.

    Example:
        >>> rnn = RNN(input_size=10, hidden_size=20, num_layers=2)
        >>> x = torch.randn(seq_len, batch_size, input_size)
        >>> output, h_n = rnn(x)

    Attributes:
        input_size: Number of expected features in input
        hidden_size: Number of features in hidden state
        num_layers: Number of recurrent layers
        nonlinearity: The non-linearity to use ('tanh' or 'relu')
        bias: If False, layer does not use bias weights
        batch_first: If True, input/output shape is (batch, seq, feature)
        dropout: Dropout probability between layers (not applied to last layer)
        bidirectional: If True, becomes a bidirectional RNN
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = 'tanh',
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Initialize RNN module.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden features
            num_layers: Number of recurrent layers
            nonlinearity: 'tanh' or 'relu'
            bias: Whether to use bias
            batch_first: If True, input is (batch, seq, feature)
            dropout: Dropout probability (0 = no dropout)
            bidirectional: If True, bidirectional RNN
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create RNN cells for each layer and direction
        self._create_cells()

    def _create_cells(self):
        """Create RNN cells for all layers."""
        raise NotImplementedError(
            "TODO: Create RNN cells for each layer\n"
            "For layer 0: input_dim = input_size\n"
            "For layer > 0: input_dim = hidden_size * num_directions\n"
            "If bidirectional, create forward and backward cells"
        )

    def forward(
        self,
        x: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process full sequence through RNN.

        Args:
            x: Input sequence (seq_len, batch, input_size) or
               (batch, seq_len, input_size) if batch_first
            h_0: Initial hidden state (num_layers * num_directions, batch, hidden_size)

        Returns:
            output: Output features (seq_len, batch, hidden_size * num_directions)
            h_n: Final hidden state (num_layers * num_directions, batch, hidden_size)
        """
        raise NotImplementedError(
            "TODO: Implement RNN forward pass\n"
            "1. Handle batch_first by transposing if needed\n"
            "2. Initialize h_0 if not provided\n"
            "3. For each layer:\n"
            "   - Process sequence through RNN cells\n"
            "   - Apply dropout between layers (except last)\n"
            "4. Return output and final hidden states"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}, bidirectional={self.bidirectional}")


class LSTM(Module):
    """
    Multi-layer LSTM module that processes full sequences.

    Wraps LSTMCell to process entire sequences and supports multiple layers.

    Example:
        >>> lstm = LSTM(input_size=10, hidden_size=20, num_layers=2)
        >>> x = torch.randn(seq_len, batch_size, input_size)
        >>> output, (h_n, c_n) = lstm(x)

    Attributes:
        input_size: Number of expected features in input
        hidden_size: Number of features in hidden state
        num_layers: Number of recurrent layers
        bias: If False, layer does not use bias weights
        batch_first: If True, input/output shape is (batch, seq, feature)
        dropout: Dropout probability between layers
        bidirectional: If True, becomes a bidirectional LSTM
        proj_size: If > 0, uses LSTM with projections
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ):
        """
        Initialize LSTM module.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden features
            num_layers: Number of recurrent layers
            bias: Whether to use bias
            batch_first: If True, input is (batch, seq, feature)
            dropout: Dropout probability (0 = no dropout)
            bidirectional: If True, bidirectional LSTM
            proj_size: If > 0, use projection to this size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.num_directions = 2 if bidirectional else 1

        # Create LSTM cells for each layer and direction
        self._create_cells()

    def _create_cells(self):
        """Create LSTM cells for all layers."""
        raise NotImplementedError(
            "TODO: Create LSTM cells for each layer\n"
            "For layer 0: input_dim = input_size\n"
            "For layer > 0: input_dim = hidden_size * num_directions\n"
            "If bidirectional, create forward and backward cells"
        )

    def forward(
        self,
        x: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Process full sequence through LSTM.

        Args:
            x: Input sequence (seq_len, batch, input_size) or
               (batch, seq_len, input_size) if batch_first
            hx: Tuple of (h_0, c_0) initial states, each of shape
                (num_layers * num_directions, batch, hidden_size)

        Returns:
            output: Output features (seq_len, batch, hidden_size * num_directions)
            (h_n, c_n): Final hidden and cell states
        """
        raise NotImplementedError(
            "TODO: Implement LSTM forward pass\n"
            "1. Handle batch_first by transposing if needed\n"
            "2. Initialize (h_0, c_0) if not provided\n"
            "3. For each layer:\n"
            "   - Process sequence through LSTM cells\n"
            "   - Apply dropout between layers (except last)\n"
            "4. Return output and final (hidden, cell) states"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}, bidirectional={self.bidirectional}")


class GRU(Module):
    """
    Multi-layer GRU module that processes full sequences.

    Wraps GRUCell to process entire sequences and supports multiple layers.

    Example:
        >>> gru = GRU(input_size=10, hidden_size=20, num_layers=2)
        >>> x = torch.randn(seq_len, batch_size, input_size)
        >>> output, h_n = gru(x)

    Attributes:
        input_size: Number of expected features in input
        hidden_size: Number of features in hidden state
        num_layers: Number of recurrent layers
        bias: If False, layer does not use bias weights
        batch_first: If True, input/output shape is (batch, seq, feature)
        dropout: Dropout probability between layers
        bidirectional: If True, becomes a bidirectional GRU
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Initialize GRU module.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden features
            num_layers: Number of recurrent layers
            bias: Whether to use bias
            batch_first: If True, input is (batch, seq, feature)
            dropout: Dropout probability (0 = no dropout)
            bidirectional: If True, bidirectional GRU
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create GRU cells for each layer and direction
        self._create_cells()

    def _create_cells(self):
        """Create GRU cells for all layers."""
        raise NotImplementedError(
            "TODO: Create GRU cells for each layer\n"
            "For layer 0: input_dim = input_size\n"
            "For layer > 0: input_dim = hidden_size * num_directions\n"
            "If bidirectional, create forward and backward cells"
        )

    def forward(
        self,
        x: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process full sequence through GRU.

        Args:
            x: Input sequence (seq_len, batch, input_size) or
               (batch, seq_len, input_size) if batch_first
            h_0: Initial hidden state (num_layers * num_directions, batch, hidden_size)

        Returns:
            output: Output features (seq_len, batch, hidden_size * num_directions)
            h_n: Final hidden state (num_layers * num_directions, batch, hidden_size)
        """
        raise NotImplementedError(
            "TODO: Implement GRU forward pass\n"
            "1. Handle batch_first by transposing if needed\n"
            "2. Initialize h_0 if not provided\n"
            "3. For each layer:\n"
            "   - Process sequence through GRU cells\n"
            "   - Apply dropout between layers (except last)\n"
            "4. Return output and final hidden states"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}, bidirectional={self.bidirectional}")


# ============================================================================
# Bidirectional Wrappers (for full sequences)
# ============================================================================

class BidirectionalRNN(Module):
    """
    Bidirectional RNN: processes sequence in both directions.

    Wrapper around RNN with bidirectional=True for convenience.
    Outputs are concatenated from both directions.

    Example:
        >>> birnn = BidirectionalRNN(input_size=10, hidden_size=20)
        >>> x = torch.randn(seq_len, batch_size, input_size)
        >>> output, h_n = birnn(x)
        >>> # output shape: (seq_len, batch_size, hidden_size * 2)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ):
        """
        Initialize bidirectional RNN.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden features per direction
            num_layers: Number of recurrent layers
            bias: Whether to use bias
            batch_first: If True, input is (batch, seq, feature)
            dropout: Dropout probability
        """
        super().__init__()
        self.rnn = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=True,
        )

    def forward(
        self,
        x: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process full sequence through bidirectional RNN.

        Args:
            x: Input sequence
            h_0: Initial hidden states for both directions

        Returns:
            output: Concatenated outputs from both directions
            h_n: Final hidden states from both directions
        """
        return self.rnn(x, h_0)


class BidirectionalLSTM(Module):
    """
    Bidirectional LSTM: processes sequence in both directions.

    Wrapper around LSTM with bidirectional=True for convenience.
    Outputs are concatenated from both directions.

    Example:
        >>> bilstm = BidirectionalLSTM(input_size=10, hidden_size=20)
        >>> x = torch.randn(seq_len, batch_size, input_size)
        >>> output, (h_n, c_n) = bilstm(x)
        >>> # output shape: (seq_len, batch_size, hidden_size * 2)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        proj_size: int = 0,
    ):
        """
        Initialize bidirectional LSTM.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden features per direction
            num_layers: Number of recurrent layers
            bias: Whether to use bias
            batch_first: If True, input is (batch, seq, feature)
            dropout: Dropout probability
            proj_size: Projection size (if > 0)
        """
        super().__init__()
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=True,
            proj_size=proj_size,
        )

    def forward(
        self,
        x: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Process full sequence through bidirectional LSTM.

        Args:
            x: Input sequence
            hx: Initial (hidden, cell) states for both directions

        Returns:
            output: Concatenated outputs from both directions
            (h_n, c_n): Final (hidden, cell) states from both directions
        """
        return self.lstm(x, hx)


class BidirectionalGRU(Module):
    """
    Bidirectional GRU: processes sequence in both directions.

    More efficient than BiLSTM due to fewer parameters while maintaining
    comparable performance for most tasks.
    """

    def __init__(self, d_in, d_h, d_out):
        """
        Initialize bidirectional GRU.

        Args:
            d_in (int): Input dimension
            d_h (int): Hidden dimension per direction
            d_out (int): Output dimension
        """
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out

        # Forward and backward GRU cells
        self.gru_forward = GRUCell(d_in, d_h, d_h)
        self.gru_backward = GRUCell(d_in, d_h, d_h)

        # Output projection for concatenated [h_f; h_b]
        init_scale = math.sqrt(2.0 / (2 * d_h + d_out))
        self.W_out = Parameter(np.random.randn(d_out, 2 * d_h) * init_scale)
        self.b_out = Parameter(np.zeros(d_out))

    def forward(self, sequence: Tensor) -> Tuple[Tensor, list, list]:
        """
        Forward pass through bidirectional GRU.

        Args:
            sequence (Tensor): Input sequence [T, batch_size, d_in]

        Returns:
            output (Tensor): Bidirectional output [T, batch_size, d_out]
            h_forward (list): Forward hidden states
            h_backward (list): Backward hidden states
        """
        raise NotImplementedError(
            "BidirectionalGRU.forward() requires implementation.\n"
            "Hints (similar to BiLSTM but with GRU):\n"
            "  1. Forward pass: process sequence left-to-right\n"
            "  2. Backward pass: process sequence right-to-left\n"
            "  3. Concatenate hidden states\n"
            "  4. Project concatenated output\n"
            "  5. Return as Tensors\n"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"d_in={self.d_in}, d_h={self.d_h}, d_out={self.d_out}"


class BidirectionalWrapper(Module):
    """
    Wrapper to make any RNN cell bidirectional.

    Can wrap RNNCell, LSTMCell, or GRUCell to create bidirectional version.
    """

    def __init__(self, cell_type, d_in, d_h, d_out, cell_kwargs=None):
        """
        Initialize bidirectional RNN wrapper.

        Args:
            cell_type (type): RNN cell class (RNNCell, LSTMCell, GRUCell)
            d_in (int): Input dimension
            d_h (int): Hidden dimension per direction
            d_out (int): Output dimension
            cell_kwargs (dict, optional): Additional kwargs for cell initialization
        """
        super().__init__()
        self.cell_type = cell_type
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out
        self.cell_kwargs = cell_kwargs or {}

        # Create forward and backward cell instances
        self.cell_forward = cell_type(d_in, d_h, d_h, **self.cell_kwargs)
        self.cell_backward = cell_type(d_in, d_h, d_h, **self.cell_kwargs)

        # Output projection for concatenated [h_f; h_b]
        init_scale = math.sqrt(2.0 / (2 * d_h + d_out))
        self.W_out = Parameter(np.random.randn(d_out, 2 * d_h) * init_scale)
        self.b_out = Parameter(np.zeros(d_out))

    def forward(self, sequence: Tensor, mask=None) -> Tuple[Tensor, list, list, dict]:
        """
        Forward pass through bidirectional RNN.

        Args:
            sequence (Tensor): Input sequence [T, batch_size, d_in]
                              or [batch_size, T, d_in]
            mask (np.ndarray, optional): Sequence length mask

        Returns:
            output (Tensor): Bidirectional output [T, batch_size, d_out]
            hidden_forward (list): Forward hidden states [T, batch_size, d_h]
            hidden_backward (list): Backward hidden states [T, batch_size, d_h]
            cache (dict): Cache for backward pass
        """
        raise NotImplementedError(
            "BidirectionalWrapper.forward() requires implementation.\n"
            "Hints:\n"
            "  1. Ensure sequence is [T, batch_size, d_in] shape\n"
            "  2. Forward pass (t=0 to T-1):\n"
            "     h_f = zeros([batch_size, d_h])\n"
            "     hidden_forward = []\n"
            "     for t in range(T):\n"
            "       y_f, h_f, _ = cell_forward.forward(sequence[t], h_f)\n"
            "       hidden_forward.append(h_f)\n"
            "  3. Backward pass (t=T-1 down to 0):\n"
            "     h_b = zeros([batch_size, d_h])\n"
            "     hidden_backward = [None] * T\n"
            "     for t in range(T-1, -1, -1):\n"
            "       y_b, h_b, _ = cell_backward.forward(sequence[t], h_b)\n"
            "       hidden_backward[t] = h_b\n"
            "  4. Concatenate and project, return as Tensors\n"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"cell_type={self.cell_type.__name__}, d_in={self.d_in}, d_h={self.d_h}, d_out={self.d_out}"


class StackedBidirectional(Module):
    """
    Multiple bidirectional RNN layers stacked vertically.

    Each layer processes output from previous layer with its own bidirectional cells.
    """

    def __init__(self, cell_type, d_in, d_h, num_layers, d_out):
        """
        Initialize stacked bidirectional RNN.

        Args:
            cell_type (type): RNN cell class (RNNCell, LSTMCell, GRUCell)
            d_in (int): Input dimension
            d_h (int): Hidden dimension per direction
            num_layers (int): Number of bidirectional layers
            d_out (int): Output dimension
        """
        super().__init__()
        self.cell_type = cell_type
        self.d_in = d_in
        self.d_h = d_h
        self.num_layers = num_layers
        self.d_out = d_out

        # Create bidirectional layer for each layer
        self.bidirectional_layers = []
        for l in range(num_layers):
            input_dim = d_in if l == 0 else 2 * d_h  # BiRNN outputs 2*d_h
            output_dim = d_out if l == num_layers - 1 else 2 * d_h
            self.bidirectional_layers.append(
                BidirectionalWrapper(cell_type, input_dim, d_h, output_dim)
            )

    def forward(self, sequence: Tensor, mask=None) -> Tuple[Tensor, list]:
        """
        Forward pass through stacked bidirectional RNN.

        Args:
            sequence (Tensor): Input sequence [T, batch_size, d_in]
            mask (np.ndarray, optional): Sequence mask

        Returns:
            output (Tensor): Output [T, batch_size, d_out]
            all_hidden_states (list): Hidden states from each layer
        """
        raise NotImplementedError(
            "StackedBidirectional.forward() requires implementation.\n"
            "Hints:\n"
            "  1. For each layer l:\n"
            "     If l == 0:\n"
            "       layer_input = sequence\n"
            "     Else:\n"
            "       layer_input = output from layer l-1\n"
            "     output, h_f, h_b, _ = bidirectional_layers[l].forward(layer_input, mask)\n"
            "  2. Stack outputs for all layers\n"
            "  3. Return as Tensors\n"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"cell_type={self.cell_type.__name__}, d_in={self.d_in}, d_h={self.d_h}, num_layers={self.num_layers}, d_out={self.d_out}"


# ============================================================================
# FUNCTIONAL CLASSES (For Computational Graph / Autograd)
# ============================================================================

class RNNCellFunction(Function):
    """
    Vanilla RNN Cell functional operation.

    Single step of a recurrent neural network.

    Math:
        h_t = tanh(x_t @ W_ih^T + b_ih + h_{t-1} @ W_hh^T + b_hh)

    Input shapes:
        x: (batch, input_size)
        h: (batch, hidden_size)
        W_ih: (hidden_size, input_size)
        W_hh: (hidden_size, hidden_size)
        b_ih, b_hh: (hidden_size,)

    Output:
        h_new: (batch, hidden_size)
    """

    def forward(
        self,
        x: np.ndarray,
        h: np.ndarray,
        W_ih: np.ndarray,
        W_hh: np.ndarray,
        b_ih: np.ndarray,
        b_hh: np.ndarray
    ) -> np.ndarray:
        """
        Compute one RNN cell step.

        Args:
            x: Input at current timestep (batch, input_size)
            h: Hidden state from previous step (batch, hidden_size)
            W_ih: Input-to-hidden weights
            W_hh: Hidden-to-hidden weights
            b_ih: Input-to-hidden bias
            b_hh: Hidden-to-hidden bias

        Returns:
            New hidden state (batch, hidden_size)
        """
        raise NotImplementedError(
            "TODO: Implement RNNCell forward\n"
            "Hint:\n"
            "  # Store for backward\n"
            "  self.x = x\n"
            "  self.h = h\n"
            "  self.W_ih = W_ih\n"
            "  self.W_hh = W_hh\n"
            "  \n"
            "  # Compute pre-activation\n"
            "  self.pre_act = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh\n"
            "  \n"
            "  # Apply activation\n"
            "  self.h_new = tanh(self.pre_act)\n"
            "  \n"
            "  return self.h_new"
        )

    def backward(
        self,
        grad_h_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for RNN cell.

        Args:
            grad_h_new: Gradient w.r.t. new hidden state

        Returns:
            Tuple of (grad_x, grad_h, grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh)
        """
        raise NotImplementedError(
            "TODO: Implement RNNCell backward\n"
            "Hint:\n"
            "  # Gradient through tanh\n"
            "  # d/dx tanh(x) = 1 - tanh(x)^2\n"
            "  grad_pre_act = grad_h_new * (1 - self.h_new ** 2)\n"
            "  \n"
            "  # Gradients w.r.t. weights and biases\n"
            "  grad_W_ih = grad_pre_act.T @ self.x\n"
            "  grad_W_hh = grad_pre_act.T @ self.h\n"
            "  grad_b_ih = np.sum(grad_pre_act, axis=0)\n"
            "  grad_b_hh = np.sum(grad_pre_act, axis=0)\n"
            "  \n"
            "  # Gradients w.r.t. inputs\n"
            "  grad_x = grad_pre_act @ self.W_ih\n"
            "  grad_h = grad_pre_act @ self.W_hh\n"
            "  \n"
            "  return grad_x, grad_h, grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh"
        )


class LSTMCellFunction(Function):
    """
    LSTM Cell functional operation.

    Single step of Long Short-Term Memory network.

    Math:
        Gates (all use sigmoid σ):
            i_t = σ(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)  (input)
            f_t = σ(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)  (forget)
            o_t = σ(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)  (output)

        Cell candidate (uses tanh):
            g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)

        State updates:
            c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
            h_t = o_t ⊙ tanh(c_t)

    Input shapes:
        x: (batch, input_size)
        h: (batch, hidden_size)
        c: (batch, hidden_size)
        W_ih: (4*hidden_size, input_size)
        W_hh: (4*hidden_size, hidden_size)
        b_ih, b_hh: (4*hidden_size,)

    Output:
        (h_new, c_new): both (batch, hidden_size)
    """

    def forward(
        self,
        x: np.ndarray,
        h: np.ndarray,
        c: np.ndarray,
        W_ih: np.ndarray,
        W_hh: np.ndarray,
        b_ih: np.ndarray,
        b_hh: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute one LSTM cell step.

        Args:
            x: Input at current timestep (batch, input_size)
            h: Hidden state from previous step (batch, hidden_size)
            c: Cell state from previous step (batch, hidden_size)
            W_ih: Input-to-hidden weights (4*hidden, input)
            W_hh: Hidden-to-hidden weights (4*hidden, hidden)
            b_ih: Input-to-hidden bias (4*hidden,)
            b_hh: Hidden-to-hidden bias (4*hidden,)

        Returns:
            Tuple of (h_new, c_new)
        """
        raise NotImplementedError(
            "TODO: Implement LSTMCell forward\n"
            "Hint:\n"
            "  # Store for backward\n"
            "  self.x = x\n"
            "  self.h = h\n"
            "  self.c = c\n"
            "  self.W_ih = W_ih\n"
            "  self.W_hh = W_hh\n"
            "  \n"
            "  hidden_size = h.shape[1]\n"
            "  \n"
            "  # Compute all gates at once\n"
            "  gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh  # (batch, 4*hidden)\n"
            "  \n"
            "  # Split into individual gates\n"
            "  i_gate = sigmoid(gates[:, 0:hidden_size])           # input gate\n"
            "  f_gate = sigmoid(gates[:, hidden_size:2*hidden_size])  # forget gate\n"
            "  g_gate = tanh(gates[:, 2*hidden_size:3*hidden_size])    # cell gate\n"
            "  o_gate = sigmoid(gates[:, 3*hidden_size:])          # output gate\n"
            "  \n"
            "  # Store gate values for backward\n"
            "  self.i_gate = i_gate\n"
            "  self.f_gate = f_gate\n"
            "  self.g_gate = g_gate\n"
            "  self.o_gate = o_gate\n"
            "  \n"
            "  # Update cell state\n"
            "  c_new = f_gate * c + i_gate * g_gate\n"
            "  self.c_new = c_new\n"
            "  \n"
            "  # Compute hidden state\n"
            "  self.tanh_c_new = tanh(c_new)\n"
            "  h_new = o_gate * self.tanh_c_new\n"
            "  \n"
            "  return h_new, c_new"
        )

    def backward(
        self,
        grad_h_new: np.ndarray,
        grad_c_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for LSTM cell.

        Args:
            grad_h_new: Gradient w.r.t. new hidden state
            grad_c_new: Gradient w.r.t. new cell state

        Returns:
            Tuple of (grad_x, grad_h, grad_c, grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh)
        """
        raise NotImplementedError(
            "TODO: Implement LSTMCell backward\n"
            "Hint:\n"
            "  hidden_size = self.h.shape[1]\n"
            "  \n"
            "  # Gradient w.r.t. cell state (from both h_new and direct c_new gradient)\n"
            "  # h_new = o_gate * tanh(c_new)\n"
            "  grad_c_total = grad_c_new + grad_h_new * self.o_gate * (1 - self.tanh_c_new ** 2)\n"
            "  \n"
            "  # Gradient w.r.t. gates\n"
            "  grad_o = grad_h_new * self.tanh_c_new * self.o_gate * (1 - self.o_gate)\n"
            "  grad_f = grad_c_total * self.c * self.f_gate * (1 - self.f_gate)\n"
            "  grad_i = grad_c_total * self.g_gate * self.i_gate * (1 - self.i_gate)\n"
            "  grad_g = grad_c_total * self.i_gate * (1 - self.g_gate ** 2)\n"
            "  \n"
            "  # Stack gate gradients\n"
            "  grad_gates = np.concatenate([grad_i, grad_f, grad_g, grad_o], axis=1)\n"
            "  \n"
            "  # Gradients w.r.t. weights and biases\n"
            "  grad_W_ih = grad_gates.T @ self.x\n"
            "  grad_W_hh = grad_gates.T @ self.h\n"
            "  grad_b_ih = np.sum(grad_gates, axis=0)\n"
            "  grad_b_hh = np.sum(grad_gates, axis=0)\n"
            "  \n"
            "  # Gradients w.r.t. inputs\n"
            "  grad_x = grad_gates @ self.W_ih\n"
            "  grad_h = grad_gates @ self.W_hh\n"
            "  grad_c = grad_c_total * self.f_gate\n"
            "  \n"
            "  return grad_x, grad_h, grad_c, grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh"
        )


class GRUCellFunction(Function):
    """
    GRU Cell functional operation.

    Single step of Gated Recurrent Unit network.

    Math:
        r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)  (reset gate)
        z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)  (update gate)
        n_t = tanh(W_in @ x_t + b_in + r_t ⊙ (W_hn @ h_{t-1} + b_hn))  (new gate)
        h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}

    Input shapes:
        x: (batch, input_size)
        h: (batch, hidden_size)
        W_ih: (3*hidden_size, input_size)
        W_hh: (3*hidden_size, hidden_size)
        b_ih, b_hh: (3*hidden_size,)

    Output:
        h_new: (batch, hidden_size)
    """

    def forward(
        self,
        x: np.ndarray,
        h: np.ndarray,
        W_ih: np.ndarray,
        W_hh: np.ndarray,
        b_ih: np.ndarray,
        b_hh: np.ndarray
    ) -> np.ndarray:
        """
        Compute one GRU cell step.

        Args:
            x: Input at current timestep (batch, input_size)
            h: Hidden state from previous step (batch, hidden_size)
            W_ih: Input-to-hidden weights (3*hidden, input)
            W_hh: Hidden-to-hidden weights (3*hidden, hidden)
            b_ih: Input-to-hidden bias (3*hidden,)
            b_hh: Hidden-to-hidden bias (3*hidden,)

        Returns:
            New hidden state (batch, hidden_size)
        """
        raise NotImplementedError(
            "TODO: Implement GRUCell forward\n"
            "Hint:\n"
            "  # Store for backward\n"
            "  self.x = x\n"
            "  self.h = h\n"
            "  self.W_ih = W_ih\n"
            "  self.W_hh = W_hh\n"
            "  \n"
            "  hidden_size = h.shape[1]\n"
            "  \n"
            "  # Compute input projections\n"
            "  gi = x @ W_ih.T + b_ih  # (batch, 3*hidden)\n"
            "  gh = h @ W_hh.T + b_hh  # (batch, 3*hidden)\n"
            "  \n"
            "  # Split projections\n"
            "  i_r, i_z, i_n = np.split(gi, 3, axis=1)\n"
            "  h_r, h_z, h_n = np.split(gh, 3, axis=1)\n"
            "  \n"
            "  # Compute gates\n"
            "  r_gate = sigmoid(i_r + h_r)  # reset gate\n"
            "  z_gate = sigmoid(i_z + h_z)  # update gate\n"
            "  n_gate = tanh(i_n + r_gate * h_n)  # new gate\n"
            "  \n"
            "  # Store for backward\n"
            "  self.r_gate = r_gate\n"
            "  self.z_gate = z_gate\n"
            "  self.n_gate = n_gate\n"
            "  self.h_n = h_n\n"
            "  \n"
            "  # Compute new hidden state\n"
            "  h_new = (1 - z_gate) * n_gate + z_gate * h\n"
            "  \n"
            "  return h_new"
        )

    def backward(
        self,
        grad_h_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for GRU cell.

        Args:
            grad_h_new: Gradient w.r.t. new hidden state

        Returns:
            Tuple of (grad_x, grad_h, grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh)
        """
        raise NotImplementedError(
            "TODO: Implement GRUCell backward\n"
            "Hint:\n"
            "  hidden_size = self.h.shape[1]\n"
            "  \n"
            "  # h_new = (1 - z) * n + z * h\n"
            "  grad_n = grad_h_new * (1 - self.z_gate)\n"
            "  grad_z = grad_h_new * (self.h - self.n_gate)\n"
            "  grad_h_direct = grad_h_new * self.z_gate\n"
            "  \n"
            "  # n = tanh(i_n + r * h_n)\n"
            "  grad_n_pre = grad_n * (1 - self.n_gate ** 2)\n"
            "  grad_r = grad_n_pre * self.h_n\n"
            "  grad_h_n = grad_n_pre * self.r_gate\n"
            "  \n"
            "  # r = sigmoid(i_r + h_r)\n"
            "  grad_r_pre = grad_r * self.r_gate * (1 - self.r_gate)\n"
            "  \n"
            "  # z = sigmoid(i_z + h_z)\n"
            "  grad_z_pre = grad_z * self.z_gate * (1 - self.z_gate)\n"
            "  \n"
            "  # Stack gate gradients [r, z, n]\n"
            "  grad_gi = np.concatenate([grad_r_pre, grad_z_pre, grad_n_pre], axis=1)\n"
            "  grad_gh = np.concatenate([grad_r_pre, grad_z_pre, grad_h_n], axis=1)\n"
            "  \n"
            "  # Gradients w.r.t. weights\n"
            "  grad_W_ih = grad_gi.T @ self.x\n"
            "  grad_W_hh = grad_gh.T @ self.h\n"
            "  grad_b_ih = np.sum(grad_gi, axis=0)\n"
            "  grad_b_hh = np.sum(grad_gh, axis=0)\n"
            "  \n"
            "  # Gradients w.r.t. inputs\n"
            "  grad_x = grad_gi @ self.W_ih\n"
            "  grad_h = grad_gh @ self.W_hh + grad_h_direct\n"
            "  \n"
            "  return grad_x, grad_h, grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh"
        )


class BidirectionalFunction(Function):
    """
    Bidirectional RNN functional operation.

    Processes sequence in both forward and backward directions,
    concatenating the outputs.

    Math:
        h_forward = RNN(x, direction='forward')
        h_backward = RNN(x, direction='backward')
        output = concat(h_forward, h_backward)
    """

    def __init__(self, cell_type: str = 'lstm'):
        """
        Initialize Bidirectional function.

        Args:
            cell_type: Type of RNN cell ('rnn', 'lstm', 'gru')
        """
        self.cell_type = cell_type
        if cell_type == 'rnn':
            self.cell_fn = RNNCellFunction
        elif cell_type == 'lstm':
            self.cell_fn = LSTMCellFunction
        elif cell_type == 'gru':
            self.cell_fn = GRUCellFunction
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

    def forward(
        self,
        x: np.ndarray,
        h_0: Optional[np.ndarray] = None,
        c_0: Optional[np.ndarray] = None,
        weights_forward: Tuple[np.ndarray, ...] = None,
        weights_backward: Tuple[np.ndarray, ...] = None
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Compute bidirectional RNN.

        Args:
            x: Input sequence (batch, seq_len, input_size)
            h_0: Initial hidden state for both directions (2, batch, hidden)
            c_0: Initial cell state (LSTM only) (2, batch, hidden)
            weights_forward: Weights for forward direction
            weights_backward: Weights for backward direction

        Returns:
            Tuple of (output, final_states)
        """
        raise NotImplementedError(
            "TODO: Implement Bidirectional forward\n"
            "Hint:\n"
            "  # Process forward direction\n"
            "  h_forward = []\n"
            "  h = h_0[0] if h_0 is not None else np.zeros(...)\n"
            "  c = c_0[0] if c_0 is not None else np.zeros(...)\n"
            "  \n"
            "  for t in range(seq_len):\n"
            "      if self.cell_type == 'lstm':\n"
            "          h, c = cell_forward.forward(x[:, t], h, c, *weights_forward)\n"
            "      else:\n"
            "          h = cell_forward.forward(x[:, t], h, *weights_forward)\n"
            "      h_forward.append(h)\n"
            "  \n"
            "  # Process backward direction (reversed sequence)\n"
            "  h_backward = []\n"
            "  h = h_0[1] if h_0 is not None else np.zeros(...)\n"
            "  \n"
            "  for t in range(seq_len - 1, -1, -1):\n"
            "      ...\n"
            "      h_backward.insert(0, h)  # Prepend to maintain order\n"
            "  \n"
            "  # Concatenate forward and backward\n"
            "  output = np.concatenate([h_forward, h_backward], axis=-1)"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Compute gradients for bidirectional RNN.

        Args:
            grad_output: Gradient w.r.t. output

        Returns:
            Tuple of gradients
        """
        raise NotImplementedError(
            "TODO: Implement Bidirectional backward\n"
            "Hint: Backprop through both directions and accumulate gradients"
        )
