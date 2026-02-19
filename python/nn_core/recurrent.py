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
from typing import Tuple, Optional, Union, Callable

from . import recurrent_functional, Dropout, Linear
from .module import Module, Parameter, Sequential
from ..foundations.functionals import Function
from ..foundations import Tensor, convert_to_function, stack, concat
from .init import xavier_normal_


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

    def __init__(self, d_in, d_h, nonlinearity: str = 'tanh', bias: bool = True):
        """
        Initialize RNN cell.

        Args:
            d_in (int): Input dimension
            d_h (int): Hidden dimension
        """
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h

        self.W_ih = Parameter(np.zeros((d_h, d_in)))
        self.W_hh = Parameter(np.zeros((d_h, d_h)))

        self.nonlinearity = nonlinearity
        if bias:
            self.b_h = Parameter(np.zeros(d_h))
        else:
            self.b_h = Tensor(np.zeros(d_h), requires_grad=False)

        self.rnn_func = convert_to_function(recurrent_functional.RNNCell)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
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
        return self.rnn_func(x_t, h_prev, self.W_ih.T, self.W_hh, self.b_h, self.nonlinearity)


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

    def __init__(self, d_in, d_h, bias=True):
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

        self.W_ih = Parameter(np.zeros((d_h * 4, d_in)))
        self.W_hh = Parameter(np.zeros((d_h * 4, d_h)))

        if bias:
            self.b_ih = Parameter(np.zeros(d_h * 4))
            self.b_hh = Parameter(np.zeros(d_h * 4))
        else:
            self.b_ih = Tensor(np.zeros(d_h * 4), requires_grad=False)
            self.b_hh = Tensor(np.zeros(d_h * 4), requires_grad=False)

        self.lstm_func = convert_to_function(recurrent_functional.LSTMCell)

    def forward(self, x_t: Tensor, h_prev: Tensor, c_prev: Tensor) -> Tuple[Tensor, Tensor]:
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
        h_new = self.lstm_func(x_t, h_prev, c_prev, self.W_ih, self.W_hh, self.b_ih, self.b_hh)
        c_new = self.lstm_func.fn.c_new
        if not isinstance(c_new, Tensor):
            c_new = Tensor(c_new, requires_grad=False)
        return h_new, c_new

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"d_in={self.d_in}, d_h={self.d_h}"


class GRUCell(Module):
    """
    Gated Recurrent Unit (GRU) Cell.

    Simplified version of LSTM with 2 gates instead of 3, and hidden state
    serving as both the state and the output.

    Uses combined weight matrices (like LSTMCell):
        W_ih: (3*d_h, d_in) - input weights for [reset, update, new] gates
        W_hh: (3*d_h, d_h)  - hidden weights for [reset, update, new] gates
        b_ih: (3*d_h,)       - input bias
        b_hh: (3*d_h,)       - hidden bias

    Attributes:
        d_in (int): Input dimension
        d_h (int): Hidden state dimension
    """

    def __init__(self, d_in, d_h, bias=True):
        """
        Initialize GRU cell.

        Args:
            d_in (int): Input dimension
            d_h (int): Hidden dimension
            bias (bool): Whether to use bias weights
        """
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h

        # Combined weight matrices: [reset_gate, update_gate, new_gate]
        self.W_ih = Parameter(np.zeros((d_h * 3, d_in)))
        self.W_hh = Parameter(np.zeros((d_h * 3, d_h)))

        if bias:
            self.b_ih = Parameter(np.zeros(d_h * 3))
            self.b_hh = Parameter(np.zeros(d_h * 3))
        else:
            self.b_ih = Tensor(np.zeros(d_h * 3), requires_grad=False)
            self.b_hh = Tensor(np.zeros(d_h * 3), requires_grad=False)

        self.gru_func = convert_to_function(recurrent_functional.GRUCell)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for single GRU time step.

        Args:
            x_t (Tensor): Input at time t [batch_size, d_in]
            h_prev (Tensor): Hidden state from previous time step [batch_size, d_h]

        Returns:
            h_new (Tensor): New hidden state [batch_size, d_h]
        """
        h_new = self.gru_func(x_t, h_prev, self.W_ih, self.W_hh, self.b_ih, self.b_hh)
        return h_new

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"d_in={self.d_in}, d_h={self.d_h}"


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
        self._init_parameters(xavier_normal_)

    def _create_cells(self):
        """Create RNN cells for all layers."""
        layers = [RNNCell(self.input_size, self.hidden_size, nonlinearity=self.nonlinearity, bias=self.bias)]
        for _ in range(self.num_layers - 1):
            layers.append(RNNCell(self.hidden_size * self.num_directions, self.hidden_size, nonlinearity=self.nonlinearity, bias=self.bias))
        self.forward_layers = layers

        if self.dropout > 1e-8:
            self.dropout_layers = [Dropout(self.dropout) for _ in range(self.num_layers - 1)]
        else:
            self.dropout_layers = None

        if self.bidirectional:
            layers = [RNNCell(self.input_size, self.hidden_size, nonlinearity=self.nonlinearity, bias=self.bias)]
            for _ in range(self.num_layers - 1):
                layers.append(RNNCell(self.hidden_size * self.num_directions, self.hidden_size, nonlinearity=self.nonlinearity, bias=self.bias))
            self.backward_layers = layers
        else:
            self.backward_layers = None


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
        if self.batch_first:
            x = x.permute(1, 0, 2)
        L, B, D = x.shape
        if h_0 is None:
            h_0 = Tensor(np.zeros((self.num_layers * self.num_directions, x.shape[1], self.hidden_size)), requires_grad=True)

        layer_outputs = x
        final_hidden = []
        for i, layer in enumerate(self.forward_layers):
            forward_outputs = [h_0[2 * i]]
            for l in range(L):
                h_new = layer(layer_outputs[l], forward_outputs[-1])
                forward_outputs.append(h_new)
            forward_outputs = forward_outputs[1:]
            final_hidden.append(forward_outputs[-1])
            if self.bidirectional:
                backward_outputs = [h_0[2 * i + 1]]
                for l in range(L - 1, -1, -1):
                    h_new = self.backward_layers[i](layer_outputs[l], backward_outputs[-1])
                    backward_outputs.append(h_new)
                backward_outputs = backward_outputs[::-1]
                backward_outputs = backward_outputs[:-1]
                final_hidden.append(backward_outputs[0])
                layer_outputs = [concat(forward_outputs[t], backward_outputs[t], axis=-1) for t in range(L)]
            else:
                layer_outputs = forward_outputs
            layer_outputs = stack(*layer_outputs)
            if self.dropout_layers is not None and i < self.num_layers - 1:
                layer_outputs = self.dropout_layers[i](layer_outputs)
        final_hidden = stack(*final_hidden)
        return layer_outputs, final_hidden


    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}, bidirectional={self.bidirectional}")


class LSTM(Module):
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
        self._create_cells()
        self._init_parameters(xavier_normal_)


    def _create_cells(self):
        in_size = self.input_size
        next_in = (self.proj_size if self.proj_size > 0 else self.hidden_size) * self.num_directions

        self.forward_layers = [LSTMCell(in_size, self.hidden_size, bias=self.bias)]
        for _ in range(self.num_layers - 1):
            self.forward_layers.append(LSTMCell(next_in, self.hidden_size, bias=self.bias))

        if self.bidirectional:
            self.backward_layers = [LSTMCell(in_size, self.hidden_size, bias=self.bias)]
            for _ in range(self.num_layers - 1):
                self.backward_layers.append(LSTMCell(next_in, self.hidden_size, bias=self.bias))
        else:
            self.backward_layers = None

        if self.dropout > 1e-8:
            self.dropout_layers = [Dropout(self.dropout) for _ in range(self.num_layers - 1)]
        else:
            self.dropout_layers = None

        if self.proj_size > 0:
            self.linear_proj = Linear(self.hidden_size, self.proj_size, bias=False)
        else:
            self.linear_proj = None

    def forward(
        self,
        x: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if self.batch_first:
            x = x.permute(1, 0, 2)
        L, B, D = x.shape

        if hx is None:
            h_0 = Tensor(np.zeros((self.num_layers * self.num_directions, B, self.hidden_size)))
            c_0 = Tensor(np.zeros((self.num_layers * self.num_directions, B, self.hidden_size)))
        else:
            h_0, c_0 = hx

        layer_outputs = x
        final_hidden = []
        final_cell = []

        for i, layer in enumerate(self.forward_layers):
            fwd_idx = i * self.num_directions

            # Forward
            h_prev, c_prev = h_0[fwd_idx], c_0[fwd_idx]
            fwd = []
            for t in range(L):
                h_prev, c_prev = layer(layer_outputs[t], h_prev, c_prev)
                if self.linear_proj is not None:
                    h_prev = self.linear_proj(h_prev)
                fwd.append((h_prev, c_prev))
            final_hidden.append(fwd[-1][0])
            final_cell.append(fwd[-1][1])

            # Backward
            if self.bidirectional:
                h_prev, c_prev = h_0[fwd_idx + 1], c_0[fwd_idx + 1]
                bwd = []
                for t in range(L - 1, -1, -1):
                    h_prev, c_prev = self.backward_layers[i](layer_outputs[t], h_prev, c_prev)
                    if self.linear_proj is not None:
                        h_prev = self.linear_proj(h_prev)
                    bwd.append((h_prev, c_prev))
                bwd = bwd[::-1]
                final_hidden.append(bwd[0][0])
                final_cell.append(bwd[0][1])
                layer_outputs = [concat(fwd[t][0], bwd[t][0], axis=-1) for t in range(L)]
            else:
                layer_outputs = [fwd[t][0] for t in range(L)]

            layer_outputs = stack(*layer_outputs)
            if self.dropout_layers is not None and i < self.num_layers - 1:
                layer_outputs = self.dropout_layers[i](layer_outputs)

        final_hidden = stack(*final_hidden)
        final_cell = stack(*final_cell)
        if self.batch_first:
            layer_outputs = layer_outputs.permute(1, 0, 2)
        return layer_outputs, (final_hidden, final_cell)

    def extra_repr(self) -> str:
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
        in_size = self.input_size
        next_in = self.hidden_size * self.num_directions

        # For layer 0: input is input_size, for subsequent layers: hidden_size * num_directions
        self.forward_layers = [GRUCell(in_size, self.hidden_size, bias=self.bias)]
        for _ in range(self.num_layers - 1):
            self.forward_layers.append(GRUCell(next_in, self.hidden_size, bias=self.bias))

        if self.bidirectional:
            self.backward_layers = [GRUCell(in_size, self.hidden_size, bias=self.bias)]
            for _ in range(self.num_layers - 1):
                self.backward_layers.append(GRUCell(next_in, self.hidden_size, bias=self.bias))
        else:
            self.backward_layers = None

        if self.dropout > 1e-8:
            self.dropout_layers = [Dropout(self.dropout) for _ in range(self.num_layers - 1)]
        else:
            self.dropout_layers = None

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
        if self.batch_first:
            x = x.permute(1, 0, 2)
        L, B, D = x.shape

        if h_0 is None:
            h_0 = Tensor(np.zeros((self.num_layers * self.num_directions, B, self.hidden_size)))

        layer_outputs = x
        final_hidden = []

        for i, layer in enumerate(self.forward_layers):
            fwd_idx = i * self.num_directions

            # Forward direction
            h_prev = h_0[fwd_idx]
            fwd = []
            for t in range(L):
                h_prev = layer(layer_outputs[t], h_prev)
                fwd.append(h_prev)
            final_hidden.append(fwd[-1])

            # Backward direction
            if self.bidirectional:
                h_prev = h_0[fwd_idx + 1]
                bwd = []
                for t in range(L - 1, -1, -1):
                    h_prev = self.backward_layers[i](layer_outputs[t], h_prev)
                    bwd.append(h_prev)
                bwd = bwd[::-1]
                final_hidden.append(bwd[0])
                layer_outputs = [concat(fwd[t], bwd[t], axis=-1) for t in range(L)]
            else:
                layer_outputs = fwd

            layer_outputs = stack(*layer_outputs)
            if self.dropout_layers is not None and i < self.num_layers - 1:
                layer_outputs = self.dropout_layers[i](layer_outputs)

        final_hidden = stack(*final_hidden)
        if self.batch_first:
            layer_outputs = layer_outputs.permute(1, 0, 2)
        return layer_outputs, final_hidden

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}, bidirectional={self.bidirectional}")
