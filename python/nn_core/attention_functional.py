"""
Attention Functional Operations
================================

This module provides functional operations for attention mechanisms.
Function classes handle the forward/backward computation with np.ndarray,
while Module classes in attention.py wrap these for Tensor operations.

Function Classes:
    - ScaledDotProductAttention: Fundamental scaled dot-product mechanism
    - MultiHeadAttention: Multi-head attention with projections
    - CrossAttention: Cross-attention for encoder-decoder
    - CausalSelfAttention: Causal (masked) self-attention

Helper Functions:
    - scaled_dot_product_attention, multi_head_attention
    - softmax, split_heads, combine_heads
"""

import math
import numpy as np
from typing import Tuple, Optional, Union

from python.foundations import Function, convert_to_function

# Global flag for gradient tracking
_no_grad = False


# =============================================================================
# Helper Functions
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax implementation.

    Args:
        x: Input array
        axis: Axis along which to normalize

    Returns:
        Softmax output with sum of 1 along the specified axis
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def split_heads(x: np.ndarray, num_heads: int, d_k: int) -> np.ndarray:
    """
    Reshape and transpose to separate attention heads.

    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        num_heads: Number of attention heads
        d_k: Dimension per head

    Returns:
        Tensor of shape [batch_size, num_heads, seq_len, d_k]
    """
    batch_size, seq_len, d_model = x.shape
    x = x.reshape(batch_size, seq_len, num_heads, d_k)
    return x.transpose(0, 2, 1, 3)


def combine_heads(x: np.ndarray, d_model: int) -> np.ndarray:
    """
    Combine attention heads back to single tensor.

    Args:
        x: Input tensor [batch_size, num_heads, seq_len, d_k]
        d_model: Total embedding dimension

    Returns:
        Tensor of shape [batch_size, seq_len, d_model]
    """
    batch_size, num_heads, seq_len, d_k = x.shape
    x = x.transpose(0, 2, 1, 3)
    return x.reshape(batch_size, seq_len, d_model)


def create_causal_mask(seq_len: int, dtype=np.float32) -> np.ndarray:
    """
    Create a causal mask matrix.

    Args:
        seq_len: Sequence length
        dtype: Data type for mask

    Returns:
        Causal mask of shape [seq_len, seq_len]
        0.0 where attention is allowed, -inf where blocked
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return np.where(mask == 0, 0.0, -1e9).astype(dtype)


# =============================================================================
# Scaled Dot-Product Attention Function Class
# =============================================================================

class ScaledDotProductAttention(Function):
    """
    Scaled Dot-Product Attention functional operation.

    The fundamental attention mechanism used in transformers.

    Math:
        scores = Q @ K^T / √d_k
        attention_weights = softmax(scores + mask)
        output = attention_weights @ V

    Input shapes:
        Q: (batch, num_heads, seq_len_q, d_k)
        K: (batch, num_heads, seq_len_k, d_k)
        V: (batch, num_heads, seq_len_k, d_v)
        mask: (batch, 1, seq_len_q, seq_len_k) or broadcastable

    Output:
        (batch, num_heads, seq_len_q, d_v)
    """

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
        dropout_p: float = 0.0,
        training: bool = True
    ) -> np.ndarray:
        """
        Compute scaled dot-product attention.

        Args:
            Q: Query tensor (batch, heads, seq_q, d_k)
            K: Key tensor (batch, heads, seq_k, d_k)
            V: Value tensor (batch, heads, seq_k, d_v)
            mask: Attention mask (optional). Use -inf for positions to mask.
            dropout_p: Dropout probability on attention weights
            training: Whether in training mode (for dropout)

        Returns:
            Attention output (batch, heads, seq_q, d_v)
        """
        raise NotImplementedError(
            "TODO: Implement ScaledDotProductAttention forward\n"
            "Hint:\n"
            "  global _no_grad\n"
            "  d_k = Q.shape[-1]\n"
            "  \n"
            "  # Compute attention scores: Q @ K^T / sqrt(d_k)\n"
            "  scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)\n"
            "  \n"
            "  # Apply mask (if provided)\n"
            "  if mask is not None:\n"
            "      scores = scores + mask  # mask should have -inf where masked\n"
            "  \n"
            "  # Softmax over keys\n"
            "  attn_weights = softmax(scores, axis=-1)\n"
            "  \n"
            "  # Apply dropout\n"
            "  if training and dropout_p > 0:\n"
            "      dropout_mask = (np.random.rand(*attn_weights.shape) > dropout_p).astype(np.float32)\n"
            "      attn_weights = attn_weights * dropout_mask / (1 - dropout_p)\n"
            "  \n"
            "  if not _no_grad:\n"
            "      self.Q = Q\n"
            "      self.K = K\n"
            "      self.V = V\n"
            "      self.attn_weights = attn_weights\n"
            "      self.dropout_p = dropout_p\n"
            "      self.training = training\n"
            "      if training and dropout_p > 0:\n"
            "          self.dropout_mask = dropout_mask\n"
            "  \n"
            "  # Compute output: attention_weights @ V\n"
            "  output = attn_weights @ V\n"
            "  \n"
            "  return output"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for scaled dot-product attention.

        Args:
            grad_output: Gradient w.r.t. output (batch, heads, seq_q, d_v)

        Returns:
            Tuple of (grad_Q, grad_K, grad_V)
        """
        raise NotImplementedError(
            "TODO: Implement ScaledDotProductAttention backward\n"
            "Hint:\n"
            "  d_k = self.Q.shape[-1]\n"
            "  \n"
            "  # Gradient w.r.t. V\n"
            "  # output = attn_weights @ V\n"
            "  # grad_V = attn_weights^T @ grad_output\n"
            "  grad_V = self.attn_weights.transpose(0, 1, 3, 2) @ grad_output\n"
            "  \n"
            "  # Gradient w.r.t. attention weights\n"
            "  grad_attn = grad_output @ self.V.transpose(0, 1, 3, 2)\n"
            "  \n"
            "  # Apply dropout gradient\n"
            "  if self.training and self.dropout_p > 0:\n"
            "      grad_attn = grad_attn * self.dropout_mask / (1 - self.dropout_p)\n"
            "  \n"
            "  # Gradient through softmax\n"
            "  sum_term = np.sum(grad_attn * self.attn_weights, axis=-1, keepdims=True)\n"
            "  grad_scores = self.attn_weights * (grad_attn - sum_term)\n"
            "  \n"
            "  # Scale gradient\n"
            "  grad_scores = grad_scores / np.sqrt(d_k)\n"
            "  \n"
            "  # Gradient w.r.t. Q and K\n"
            "  grad_Q = grad_scores @ self.K\n"
            "  grad_K = grad_scores.transpose(0, 1, 3, 2) @ self.Q\n"
            "  \n"
            "  return grad_Q, grad_K, grad_V"
        )


# =============================================================================
# Multi-Head Attention Function Class
# =============================================================================

class MultiHeadAttention(Function):
    """
    Multi-Head Attention functional operation.

    Projects Q, K, V into multiple heads, applies attention, and combines.

    Math:
        head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
        output = Concat(head_1, ..., head_h) @ W_O

    Input shapes:
        Q: (batch, seq_len_q, d_model)
        K: (batch, seq_len_k, d_model)
        V: (batch, seq_len_k, d_model)
        W_Q, W_K, W_V: (d_model, d_model)
        W_O: (d_model, d_model)
    """

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
        W_O: np.ndarray,
        num_heads: int,
        mask: Optional[np.ndarray] = None,
        dropout_p: float = 0.0,
        training: bool = True
    ) -> np.ndarray:
        """
        Compute multi-head attention.

        Args:
            Q: Query tensor (batch, seq_q, d_model)
            K: Key tensor (batch, seq_k, d_model)
            V: Value tensor (batch, seq_k, d_model)
            W_Q: Query projection weights (d_model, d_model)
            W_K: Key projection weights (d_model, d_model)
            W_V: Value projection weights (d_model, d_model)
            W_O: Output projection weights (d_model, d_model)
            num_heads: Number of attention heads
            mask: Attention mask (optional)
            dropout_p: Dropout probability
            training: Whether in training mode

        Returns:
            Output tensor (batch, seq_q, d_model)
        """
        raise NotImplementedError(
            "TODO: Implement MultiHeadAttention forward\n"
            "Hint:\n"
            "  global _no_grad\n"
            "  \n"
            "  batch_size, seq_len_q, d_model = Q.shape\n"
            "  _, seq_len_k, _ = K.shape\n"
            "  d_k = d_model // num_heads\n"
            "  \n"
            "  # Project Q, K, V\n"
            "  Q_proj = Q @ W_Q  # (batch, seq_q, d_model)\n"
            "  K_proj = K @ W_K  # (batch, seq_k, d_model)\n"
            "  V_proj = V @ W_V  # (batch, seq_k, d_model)\n"
            "  \n"
            "  # Reshape to (batch, num_heads, seq, d_k)\n"
            "  Q_heads = split_heads(Q_proj, num_heads, d_k)\n"
            "  K_heads = split_heads(K_proj, num_heads, d_k)\n"
            "  V_heads = split_heads(V_proj, num_heads, d_k)\n"
            "  \n"
            "  # Apply scaled dot-product attention\n"
            "  attn_fn = ScaledDotProductAttention()\n"
            "  attn_output = attn_fn.forward(Q_heads, K_heads, V_heads, mask, dropout_p, training)\n"
            "  \n"
            "  # Reshape back: (batch, seq_q, d_model)\n"
            "  concat_output = combine_heads(attn_output, d_model)\n"
            "  \n"
            "  # Final projection\n"
            "  output = concat_output @ W_O\n"
            "  \n"
            "  if not _no_grad:\n"
            "      self.Q = Q\n"
            "      self.K = K\n"
            "      self.V = V\n"
            "      self.W_Q = W_Q\n"
            "      self.W_K = W_K\n"
            "      self.W_V = W_V\n"
            "      self.W_O = W_O\n"
            "      self.num_heads = num_heads\n"
            "      self.Q_heads = Q_heads\n"
            "      self.K_heads = K_heads\n"
            "      self.V_heads = V_heads\n"
            "      self.attn_output = attn_output\n"
            "      self.concat_output = concat_output\n"
            "      self.attn_fn = attn_fn\n"
            "  \n"
            "  return output"
        )

    def backward(
        self,
        grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for multi-head attention.

        Args:
            grad_output: Gradient w.r.t. output (batch, seq_q, d_model)

        Returns:
            Tuple of (grad_Q, grad_K, grad_V, grad_W_Q, grad_W_K, grad_W_V, grad_W_O)
        """
        raise NotImplementedError(
            "TODO: Implement MultiHeadAttention backward\n"
            "Hint:\n"
            "  batch_size, seq_len_q, d_model = self.Q.shape\n"
            "  _, seq_len_k, _ = self.K.shape\n"
            "  d_k = d_model // self.num_heads\n"
            "  \n"
            "  # Gradient w.r.t. W_O\n"
            "  grad_W_O = self.concat_output.reshape(-1, d_model).T @ grad_output.reshape(-1, d_model)\n"
            "  \n"
            "  # Gradient w.r.t. concat_output\n"
            "  grad_concat = grad_output @ self.W_O.T\n"
            "  \n"
            "  # Reshape to attention output shape\n"
            "  grad_attn_output = split_heads(grad_concat, self.num_heads, d_k)\n"
            "  \n"
            "  # Gradient through attention\n"
            "  grad_Q_heads, grad_K_heads, grad_V_heads = self.attn_fn.backward(grad_attn_output)\n"
            "  \n"
            "  # Reshape heads back\n"
            "  grad_Q_proj = combine_heads(grad_Q_heads, d_model)\n"
            "  grad_K_proj = combine_heads(grad_K_heads, d_model)\n"
            "  grad_V_proj = combine_heads(grad_V_heads, d_model)\n"
            "  \n"
            "  # Gradient w.r.t. projection weights\n"
            "  grad_W_Q = self.Q.reshape(-1, d_model).T @ grad_Q_proj.reshape(-1, d_model)\n"
            "  grad_W_K = self.K.reshape(-1, d_model).T @ grad_K_proj.reshape(-1, d_model)\n"
            "  grad_W_V = self.V.reshape(-1, d_model).T @ grad_V_proj.reshape(-1, d_model)\n"
            "  \n"
            "  # Gradient w.r.t. inputs\n"
            "  grad_Q = grad_Q_proj @ self.W_Q.T\n"
            "  grad_K = grad_K_proj @ self.W_K.T\n"
            "  grad_V = grad_V_proj @ self.W_V.T\n"
            "  \n"
            "  return grad_Q, grad_K, grad_V, grad_W_Q, grad_W_K, grad_W_V, grad_W_O"
        )


# =============================================================================
# Cross-Attention Function Class
# =============================================================================

class CrossAttention(Function):
    """
    Cross-Attention functional operation.

    Attention where queries come from one sequence and keys/values from another.
    Used in encoder-decoder architectures.

    Input shapes:
        Q: (batch, seq_len_q, d_model) - from decoder
        K, V: (batch, seq_len_k, d_model) - from encoder
    """

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
        W_O: np.ndarray,
        num_heads: int,
        mask: Optional[np.ndarray] = None,
        dropout_p: float = 0.0,
        training: bool = True
    ) -> np.ndarray:
        """
        Compute cross-attention.

        Args:
            Q: Query tensor from decoder (batch, seq_q, d_model)
            K: Key tensor from encoder (batch, seq_k, d_model)
            V: Value tensor from encoder (batch, seq_k, d_model)
            W_Q, W_K, W_V, W_O: Projection weights
            num_heads: Number of attention heads
            mask: Optional mask for encoder padding
            dropout_p: Dropout probability
            training: Whether in training mode

        Returns:
            Output tensor (batch, seq_q, d_model)
        """
        raise NotImplementedError(
            "TODO: Implement CrossAttention forward\n"
            "Hint: Same as MultiHeadAttention but Q comes from decoder, K/V from encoder"
        )

    def backward(
        self,
        grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients for cross-attention."""
        raise NotImplementedError("TODO: Implement CrossAttention backward")


# =============================================================================
# Causal Self-Attention Function Class
# =============================================================================

class CausalSelfAttention(Function):
    """
    Causal (Masked) Self-Attention functional operation.

    Self-attention with causal masking to prevent attending to future positions.
    Used in autoregressive models like GPT.

    The mask ensures position i can only attend to positions 0...i.
    """

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
        W_O: np.ndarray,
        num_heads: int,
        dropout_p: float = 0.0,
        training: bool = True
    ) -> np.ndarray:
        """
        Compute causal self-attention.

        Args:
            Q, K, V: Input tensors (usually the same for self-attention)
            W_Q, W_K, W_V, W_O: Projection weights
            num_heads: Number of attention heads
            dropout_p: Dropout probability
            training: Whether in training mode

        Returns:
            Output tensor
        """
        raise NotImplementedError(
            "TODO: Implement CausalSelfAttention forward\n"
            "Hint:\n"
            "  seq_len = Q.shape[1]\n"
            "  \n"
            "  # Create causal mask: upper triangular with -inf\n"
            "  causal_mask = create_causal_mask(seq_len)\n"
            "  causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq, seq)\n"
            "  \n"
            "  # Use MultiHeadAttention with causal mask\n"
            "  mha = MultiHeadAttention()\n"
            "  return mha.forward(Q, K, V, W_Q, W_K, W_V, W_O, num_heads, causal_mask, dropout_p, training)"
        )

    def backward(
        self,
        grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients for causal self-attention."""
        raise NotImplementedError("TODO: Implement CausalSelfAttention backward")


# =============================================================================
# Functional Interfaces
# =============================================================================

scaled_dot_product_attention = convert_to_function(ScaledDotProductAttention)
multi_head_attention = convert_to_function(MultiHeadAttention)
cross_attention = convert_to_function(CrossAttention)
causal_self_attention = convert_to_function(CausalSelfAttention)
