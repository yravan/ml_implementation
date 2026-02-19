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
from python.utils.math_utils import sigmoid


# =============================================================================
# RNN Cell Function Class
# =============================================================================

class RNNCell(Function):
    """
    Vanilla RNN Cell functional operation.

    Single step of a recurrent neural network.

    Math:
        h_t = tanh(x_t @ W_ih + b_ih + h_{t-1} @ W_hh + b_hh)

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
            W_ih: Input-to-hidden weights (input_size, hidden_size)
            W_hh: Hidden-to-hidden weights (hidden_size, hidden_size)
            b_ih: Input-to-hidden bias
            b_hh: Hidden-to-hidden bias (hidden_size)

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
            if 'x' not in self.__dict__:
                self.x = []
            self.x.append(x)
            if 'h' not in self.__dict__:
                self.h = []
            self.h.append(h)
            self.h_new = h_new
            self.W_ih = W_ih
            self.W_hh = W_hh
            self.nonlinearity = nonlinearity
        return h_new

    def backward(
        self,
        grad_h_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for RNN cell.

        Args:
            grad_h_new: Gradient w.r.t. new hidden state (batch, hidden_size)

        Returns:
            Tuple of (grad_x, grad_h, grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh)
        """
        if self.nonlinearity == 'tanh':
            grad_pre_act = (1 - self.h_new ** 2) * grad_h_new
        elif self.nonlinearity == 'relu':
            grad_pre_act = (self.h_new > 0).astype(grad_h_new.dtype) * grad_h_new
        else:
            raise ValueError("Nonlinearity not recognized", self.nonlinearity)

        grad_b_h = grad_pre_act.sum(axis=0) # hidden_size

        grad_W_ih = self.x[-1].T @ grad_pre_act # input_size, hidden_size
        grad_W_hh = self.h[-1].T @ grad_pre_act # hidden_size, hidden_size

        grad_x = grad_pre_act @ self.W_ih.T # batch, input_size
        grad_h = grad_pre_act @ self.W_hh.T # batch, hidden_size

        self.h_new = self.h.pop()
        self.x.pop()

        return grad_x, grad_h, grad_W_ih, grad_W_hh, grad_b_h


# =============================================================================
# LSTM Cell Function Class
# =============================================================================
class LSTMCell(Function):

    def forward(
        self,
        x: np.ndarray,
        h: np.ndarray,
        c: np.ndarray,
        W_ih: np.ndarray,
        W_hh: np.ndarray,
        b_ih: np.ndarray,
        b_hh: np.ndarray
    ) -> np.ndarray:
        gates = x @ W_ih.T + h @ W_hh.T + b_ih + b_hh  # (batch, 4*hidden)
        H = h.shape[-1]
        i = sigmoid(gates[:, :H])
        f = sigmoid(gates[:, H:2*H])
        g = np.tanh(gates[:, 2*H:3*H])
        o = sigmoid(gates[:, 3*H:])

        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)

        global _no_grad
        if not _no_grad:
            if 'xs' not in self.__dict__:
                self.xs = []
                self.hs = []
                self.cs = []
                self.i_gates = []
                self.f_gates = []
                self.g_gates = []
                self.o_gates = []
                self.c_news = []
            self.xs.append(x)
            self.hs.append(h)
            self.cs.append(c)
            self.i_gates.append(i)
            self.f_gates.append(f)
            self.g_gates.append(g)
            self.o_gates.append(o)
            self.c_news.append(c_new)
            self.W_ih = W_ih
            self.W_hh = W_hh
            self.grad_c_carried = np.zeros_like(c)

        self.c_new = c_new
        return h_new

    def backward(
        self,
        grad_h_new: np.ndarray,
    ):
        x = self.xs.pop()
        h = self.hs.pop()
        c = self.cs.pop()
        i = self.i_gates.pop()
        f = self.f_gates.pop()
        g = self.g_gates.pop()
        o = self.o_gates.pop()
        c_new = self.c_news.pop()

        tanh_c_new = np.tanh(c_new)

        # Gradient through h = o * tanh(c_new)
        grad_o = tanh_c_new * grad_h_new
        grad_c = o * (1 - tanh_c_new ** 2) * grad_h_new + self.grad_c_carried

        # Gradient through c_new = f * c + i * g
        grad_f = c * grad_c
        grad_i = g * grad_c
        grad_g = i * grad_c

        # Carry cell gradient backward in time
        self.grad_c_carried = f * grad_c

        # Gradient through gate activations
        grad_gates = np.concatenate([
            grad_i * i * (1 - i),       # sigmoid'
            grad_f * f * (1 - f),       # sigmoid'
            grad_g * (1 - g ** 2),      # tanh'
            grad_o * o * (1 - o),       # sigmoid'
        ], axis=-1)  # (batch, 4*hidden)

        grad_b_ih = grad_gates.sum(axis=0)
        grad_b_hh = grad_gates.sum(axis=0)

        grad_W_ih = grad_gates.T @ x
        grad_W_hh = grad_gates.T @ h

        grad_x = grad_gates @ self.W_ih
        grad_h = grad_gates @ self.W_hh

        return grad_x, grad_h, grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh
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
        global _no_grad
        hidden_size = h.shape[1]

        # Compute input and hidden projections
        gi = x @ W_ih.T + b_ih  # (batch, 3*hidden)
        gh = h @ W_hh.T + b_hh  # (batch, 3*hidden)

        # Split projections into gate components
        i_r, i_z, i_n = np.split(gi, 3, axis=1)
        h_r, h_z, h_n = np.split(gh, 3, axis=1)

        # Compute gates
        r_gate = sigmoid(i_r + h_r)  # reset gate
        z_gate = sigmoid(i_z + h_z)  # update gate
        n_gate = np.tanh(i_n + r_gate * h_n)  # new gate (candidate)

        # Compute new hidden state via interpolation
        h_new = (1 - z_gate) * n_gate + z_gate * h

        if not _no_grad:
            self.x = x
            self.h = h
            self.W_ih = W_ih
            self.W_hh = W_hh
            self.r_gate = r_gate
            self.z_gate = z_gate
            self.n_gate = n_gate
            self.h_n = h_n

        return h_new

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
        # h_new = (1 - z) * n + z * h
        grad_n = grad_h_new * (1 - self.z_gate)
        grad_z = grad_h_new * (self.h - self.n_gate)
        grad_h_direct = grad_h_new * self.z_gate

        # n = tanh(i_n + r * h_n)
        grad_n_pre = grad_n * (1 - self.n_gate ** 2)
        grad_r = grad_n_pre * self.h_n
        grad_h_n = grad_n_pre * self.r_gate

        # r = sigmoid(i_r + h_r)
        grad_r_pre = grad_r * self.r_gate * (1 - self.r_gate)

        # z = sigmoid(i_z + h_z)
        grad_z_pre = grad_z * self.z_gate * (1 - self.z_gate)

        # Stack gate gradients [r, z, n]
        grad_gi = np.concatenate([grad_r_pre, grad_z_pre, grad_n_pre], axis=1)
        grad_gh = np.concatenate([grad_r_pre, grad_z_pre, grad_h_n], axis=1)

        # Gradients w.r.t. weights
        grad_W_ih = grad_gi.T @ self.x
        grad_W_hh = grad_gh.T @ self.h
        grad_b_ih = np.sum(grad_gi, axis=0)
        grad_b_hh = np.sum(grad_gh, axis=0)

        # Gradients w.r.t. inputs
        grad_x = grad_gi @ self.W_ih
        grad_h = grad_gh @ self.W_hh + grad_h_direct

        return grad_x, grad_h, grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh

