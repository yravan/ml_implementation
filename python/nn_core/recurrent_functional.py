"""
Recurrent Functional Operations
================================

This module provides functional operations for recurrent layers.
Function classes handle the forward/backward computation with np.ndarray,
while Module classes in recurrent.py wrap these for Tensor operations.

Function Classes:
    - RNNCell: Vanilla RNN cell functional
    - LSTMCell: LSTM cell functional
    - GRUCell: GRU cell functional
    - Bidirectional: Bidirectional RNN wrapper functional

Helper Functions:
    - rnn_cell, lstm_cell, gru_cell: Functional interfaces
    - sigmoid, tanh: Activation helpers
"""

import numpy as np
from typing import Tuple, Optional, Union

from python.foundations import Function, convert_to_function, _no_grad
from python.nn_core.activations_functional import tanh


# =============================================================================
# RNN Cell Function Class
# =============================================================================

class RNNCell(Function):
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
        b_h: np.ndarray,
        nonlinearity: str = 'tanh',
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
        pre_act = x @ W_ih + h @ W_hh + b_h
        if nonlinearity == 'tanh':
            h_new = np.tanh(pre_act)
        elif nonlinearity == 'relu':
            h_new =  np.maximum(0, pre_act)
        else:
            raise ValueError('Nonlinearity not recognized', nonlinearity)
        global _no_grad
        if not _no_grad:
            self.x = x
            self.W_ih = W_ih
            self.h = h
            self.W_hh = W_hh
            self.nonlinearity = nonlinearity
            self.h_new = h_new
        return h_new

    def backward(
        self,
        grad_h_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for RNN cell.

        Args:
            grad_h_new: Gradient w.r.t. new hidden state

        Returns:
            Tuple of (grad_x, grad_h, grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh)
        """
        if self.nonlinearity == 'tanh':
            grad_pre_act = (1 - self.h_new ** 2) * grad_h_new
        if self.nonlinearity == 'relu':
            grad_pre_act = (self.h_new > 0).astype(grad_h_new.dtype) * grad_h_new

        grad_b_h = grad_pre_act.sum(axis=0)

        grad_W_ih = self.x.T @ grad_h_new
        grad_W_hh = self.h.T @ grad_h_new

        grad_x = (self.W_ih @ grad_h_new).T
        grad_h = (self.W_hh @ grad_h_new).T

        return grad_x, grad_h, grad_W_ih, grad_W_hh, grad_b_h


# =============================================================================
# LSTM Cell Function Class
# =============================================================================

class LSTMCell(Function):
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
            "  global _no_grad\n"
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
            "  # Update cell state\n"
            "  c_new = f_gate * c + i_gate * g_gate\n"
            "  \n"
            "  # Compute hidden state\n"
            "  tanh_c_new = tanh(c_new)\n"
            "  h_new = o_gate * tanh_c_new\n"
            "  \n"
            "  if not _no_grad:\n"
            "      self.x = x\n"
            "      self.h = h\n"
            "      self.c = c\n"
            "      self.W_ih = W_ih\n"
            "      self.W_hh = W_hh\n"
            "      self.i_gate = i_gate\n"
            "      self.f_gate = f_gate\n"
            "      self.g_gate = g_gate\n"
            "      self.o_gate = o_gate\n"
            "      self.c_new = c_new\n"
            "      self.tanh_c_new = tanh_c_new\n"
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


# =============================================================================
# GRU Cell Function Class
# =============================================================================

class GRUCell(Function):
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
            "  global _no_grad\n"
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
            "  # Compute new hidden state\n"
            "  h_new = (1 - z_gate) * n_gate + z_gate * h\n"
            "  \n"
            "  if not _no_grad:\n"
            "      self.x = x\n"
            "      self.h = h\n"
            "      self.W_ih = W_ih\n"
            "      self.W_hh = W_hh\n"
            "      self.r_gate = r_gate\n"
            "      self.z_gate = z_gate\n"
            "      self.n_gate = n_gate\n"
            "      self.h_n = h_n\n"
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


# =============================================================================
# Bidirectional Function Class
# =============================================================================

class Bidirectional(Function):
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
            self.cell_fn = RNNCell
        elif cell_type == 'lstm':
            self.cell_fn = LSTMCell
        elif cell_type == 'gru':
            self.cell_fn = GRUCell
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
