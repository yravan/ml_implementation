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
from python.utils.math_utils import softmax

# Global flag for gradient tracking
_no_grad = False


# =============================================================================
# Helper Functions
# =============================================================================
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
        d_k = Q.shape[-1]
        attn_matrix = softmax(Q @ K.T / np.sqrt(d_k) + mask, axis=-1)
        if training and dropout_p > 1e-8:
            dropout_mask = np.random.uniform(size=attn_matrix.shape) > dropout_p
            attn_output = (attn_matrix * dropout_mask.astype(attn_matrix.dtype) * 1 / (1 - dropout_p)) @ V
        else:
            attn_output = (attn_matrix @ V)
            dropout_mask = None

        global _no_grad
        if not _no_grad:
            self.attn_matrix = attn_matrix
            self.V = V
            self.K = K
            self.Q = Q
            self.d_k = d_k
            self.dropout_mask = dropout_mask
            self.dropout_p = dropout_p
            self.attn_mask = mask

        return attn_output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for scaled dot-product attention.

        Args:
            grad_output: Gradient w.r.t. output (batch, heads, seq_q, d_v)

        Returns:
            Tuple of (grad_Q, grad_K, grad_V)
        """
        grad_V = self.attn_matrix.T @ grad_output
        grad_attn_matrix = grad_output @ self.V.T
        if self.dropout_mask is not None:
            grad_attn_matrix = grad_attn_matrix * self.dropout_mask / (1 - self.dropout_p)
        weighted_sum = (grad_attn_matrix * self.attn_matrix).sum(axis=-1, keepdims=True)
        grad_pre_softmax = self.attn_matrix * (grad_attn_matrix - weighted_sum) / np.sqrt(self.d_k)
        grad_Q = grad_pre_softmax @ self.K
        grad_K = grad_pre_softmax.T @ self.Q

        return grad_Q, grad_K, grad_V

# =============================================================================
# Functional Interfaces
# =============================================================================

scaled_dot_product_attention = convert_to_function(ScaledDotProductAttention)
