"""
nn_core.attention - Attention Mechanisms

This module provides attention operations for deep learning,
including Scaled Dot-Product Attention, Multi-Head Attention,
Cross-Attention, and Causal Self-Attention.

Classes:
    - ScaledDotProductAttention: Fundamental scaled dot-product mechanism
    - MultiHeadAttention: Multi-head attention with learnable projections
    - CrossAttention: Cross-attention for encoder-decoder models
    - CachedCrossAttention: Cross-attention with KV caching for efficient decoding
    - MultimodalCrossAttention: Cross-attention for multimodal inputs
    - CausalMask: Utilities for causal masking in autoregressive models
    - MultiQueryAttention: Shared key-value across query heads
    - GroupedQueryAttention: Configurable KV groups for memory-quality trade-off

Functional Operations:
    - ScaledDotProductAttentionFunction: Stateful function for autograd
    - MultiHeadAttentionFunction: Stateful function for multi-head attention
    - CrossAttentionFunction: Stateful function for cross-attention
    - CausalSelfAttentionFunction: Stateful function for causal self-attention
"""

import math
import numpy as np
from typing import Tuple, Optional, Union, Dict, List
from abc import ABC, abstractmethod

from . import attention_functional
from .module import Module, Parameter
from ..foundations import convert_to_function, concat
from ..foundations.functionals import Function


# ============================================================================
# Utility Functions
# ============================================================================

def split_heads(x: np.ndarray, num_heads: int, d_k: int) -> np.ndarray:
    """
    Reshape and transpose to separate attention heads.

    Args:
        x (np.ndarray): Input tensor of shape [batch_size, seq_len, d_model]
        num_heads (int): Number of attention heads
        d_k (int): Dimension per head

    Returns:
        np.ndarray: Tensor of shape [batch_size, num_heads, seq_len, d_k]

    Algorithm:
        1. Reshape from [B, L, d_model] to [B, L, num_heads, d_k]
        2. Transpose axes to [B, num_heads, L, d_k] for efficient computation
    """
    batch_size, seq_len, d_model = x.shape
    x = x.reshape(batch_size, seq_len, num_heads, d_k)
    return x.transpose(0, 2, 1, 3)


def combine_heads(x: np.ndarray, d_model: int) -> np.ndarray:
    """
    Combine attention heads back to single tensor.

    Args:
        x (np.ndarray): Input tensor of shape [batch_size, num_heads, seq_len, d_k]
        d_model (int): Total embedding dimension

    Returns:
        np.ndarray: Tensor of shape [batch_size, seq_len, d_model]

    Algorithm:
        1. Transpose axes from [B, num_heads, L, d_k] to [B, L, num_heads, d_k]
        2. Reshape to [B, L, d_model]
    """
    batch_size, num_heads, seq_len, d_k = x.shape
    x = x.transpose(0, 2, 1, 3)
    return x.reshape(batch_size, seq_len, d_model)


# ============================================================================
# Scaled Dot-Product Attention
# ============================================================================

class ScaledDotProductAttention(Module):
    """
    Scaled Dot-Product Attention mechanism.

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    https://arxiv.org/abs/1706.03762

    This module implements the fundamental attention mechanism:

    ATTENTION EQUATION:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Attributes:
        dropout_p (float): Dropout probability applied to attention weights. Default: 0.0
    """

    def __init__(self, dropout_p: float = 0.0):
        """
        Initialize scaled dot-product attention.

        Args:
            dropout_p (float): Dropout probability. Default: 0.0
        """
        super().__init__()
        self.dropout_p = dropout_p
        self.fn = convert_to_function(attention_functional.ScaledDotProductAttention)

    def extra_repr(self) -> str:
        """Extra representation for printing module details."""
        return f"dropout_p={self.dropout_p}"

    def forward(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention.

        Args:
            query (Tensor): Query tensor of shape [batch_size, seq_len_q, d_k]
            key (Tensor): Key tensor of shape [batch_size, seq_len_k, d_k]
            value (Tensor): Value tensor of shape [batch_size, seq_len_v, d_v]
            mask (Tensor, optional): Mask tensor of shape [batch_size, seq_len_q, seq_len_k].
                                    Boolean mask where True means "attend" (not masked).
                                    Or float mask where -inf means "don't attend".
                                    Default: None (no masking)
            training (bool): Whether in training mode (applies dropout). Default: False

        Returns:
            output (Tensor): Attention output of shape [batch_size, seq_len_q, d_v]
            attention_weights (Tensor): Softmax attention weights of shape
                                       [batch_size, seq_len_q, seq_len_k]

        Shape Notes:
            - batch_size: Number of parallel sequences in the batch
            - seq_len_q: Query sequence length (usually target sequence length)
            - seq_len_k: Key/value sequence length (usually source sequence length)
            - d_k: Dimension of key/query (typically d_model // num_heads)
            - d_v: Dimension of value (typically d_model // num_heads)

        Algorithm:
            1. Compute attention scores: scores = Q @ K^T / sqrt(d_k)
               Shape: [batch_size, seq_len_q, seq_len_k]
            2. Apply mask if provided
            3. Apply softmax over last dimension (seq_len_k)
            4. Apply dropout if training
            5. Compute output: output = weights @ V
               Shape: [batch_size, seq_len_q, d_v]
        """
        return self.fn(query, key, value, mask=mask, dropout_p=self.dropout_p, training=self.training)


# ============================================================================
# Multi-Head Attention
# ============================================================================

class MultiHeadAttention(Module):
    """
    Multi-Head Attention mechanism with learnable projections.

    Also supports Cross Attention

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    https://arxiv.org/abs/1706.03762

    The module projects Q, K, V to multiple subspaces, applies attention
    independently in each subspace, and concatenates the results.

    Attributes:
        d_model (int): Total embedding dimension
        num_heads (int): Number of attention heads
        d_k (int): Dimension of key/query per head (d_model // num_heads)
        d_v (int): Dimension of value per head (d_model // num_heads)
        dropout_p (float): Dropout probability on attention weights
        W_q (Parameter): Query projection weights [d_model, d_model]
        W_k (Parameter): Key projection weights [d_model, d_model]
        W_v (Parameter): Value projection weights [d_model, d_model]
        W_o (Parameter): Output projection weights [d_model, d_model]
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        """
        Initialize multi-head attention.

        Args:
            d_model (int): Embedding dimension. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            dropout_p (float): Dropout probability on attention weights. Default: 0.0

        Raises:
            AssertionError: If d_model is not divisible by num_heads
        """
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.dropout_p = dropout_p

        # Initialize projection weights (using He initialization)
        # All projections are [d_model, d_model]
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.W_q = Parameter(np.random.randn(d_model, d_model) * scale)
        self.W_k = Parameter(np.random.randn(d_model, d_model) * scale)
        self.W_v = Parameter(np.random.randn(d_model, d_model) * scale)
        self.W_o = Parameter(np.random.randn(d_model, d_model) * scale)

        # For scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout_p=dropout_p)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            query (Tensor): Query tensor of shape [batch_size, seq_len_q, d_model]
            key (Tensor): Key tensor of shape [batch_size, seq_len_k, d_model]
            value (Tensor): Value tensor of shape [batch_size, seq_len_v, d_model]
            mask (Tensor, optional): Attention mask of shape:
                                    - [batch_size, seq_len_q, seq_len_k] for standard mask
                                    - [batch_size, 1, seq_len_k] for broadcasted mask
                                    Default: None (no masking)
            training (bool): Whether in training mode. Default: False

        Returns:
            output (Tensor): Attention output of shape [batch_size, seq_len_q, d_model]
            attention_weights (dict): Dictionary containing attention weights for each head:
                                     {head_0: [...], head_1: [...], ...}
                                     Each entry has shape [batch_size, seq_len_q, seq_len_k]

        Algorithm:
            1. Project inputs to all heads:
               Q_proj = query @ W_q  [B, L_q, d_model]
               K_proj = key @ W_k    [B, L_k, d_model]
               V_proj = value @ W_v  [B, L_v, d_model]

            2. Reshape to separate heads:
               Q_heads = reshape(Q_proj, [B, L_q, h, d_k]) -> [B, h, L_q, d_k]
               K_heads = reshape(K_proj, [B, L_k, h, d_k]) -> [B, h, L_k, d_k]
               V_heads = reshape(V_proj, [B, L_v, h, d_k]) -> [B, h, L_v, d_k]

            3. Apply scaled dot-product attention per head:
               head_output_i = Attention(Q_heads[:, i], K_heads[:, i], V_heads[:, i])

            4. Concatenate head outputs:
               concat = reshape([head_0, ..., head_h], [B, L_q, d_model])

            5. Apply output projection:
               output = concat @ W_o

        Shape Transformations:
            Input:  [B, L, d_model]
            After projection:  [B, L, d_model]
            After reshape: [B, L, h, d_k] -> need to transpose to [B, h, L, d_k]
            After attention: [B, h, L, d_k]
            After concat: [B, L, d_model]
            Output: [B, L, d_model]
        """
        query = split_heads(query @ self.W_q, self.num_heads, self.d_k)
        key = split_heads(key @ self.W_k, self.num_heads, self.d_k)
        value = split_heads(value @ self.W_v, self.num_heads, self.d_k)
        output = self.attention(query, key, value, mask=mask)
        output = combine_heads(output, self.d_model)
        return output @ self.W_o

    def extra_repr(self) -> str:
        """Extra representation for printing module details."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, dropout_p={self.dropout_p}"


class CachedMultiHeadAttention(MultiHeadAttention):
    """
    Cross-Attention with K/V caching for efficient decoding.

    Caches encoder K/V projections to avoid recomputation during beam search
    or greedy decoding. Only recomputes as tgt_len grows.

    Attributes:
        Same as CrossAttention, plus:
        - cached_K (np.ndarray): Cached key projection [batch_size, num_heads, src_len, d_k]
        - cached_V (np.ndarray): Cached value projection [batch_size, num_heads, src_len, d_v]
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        """Initialize cached cross-attention."""
        super().__init__(d_model, num_heads, dropout_p)
        self.cached_K = None
        self.cached_V = None
        self.encoder_output_cached = None

    def extra_repr(self) -> str:
        """Extra representation for printing module details."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, dropout_p={self.dropout_p}, cached=True"

    def forward(self, query, key=None, value=None, mask=None, use_cache=True):
        """
        Forward pass with optional K/V caching.

        Args:
            query (Tensor): Decoder query [batch_size, tgt_len, d_model]
            key (Tensor): Encoder key [batch_size, tgt_len or 1, d_model]
            value (Tensor): Encoder value [batch_size, tgt_len or 1, d_model]
            mask (Tensor, optional): Encoder padding mask
            training (bool): Whether in training mode
            use_cache (bool): Whether to use/create K/V cache. Default: True

        Returns:
            output (Tensor): Cross-attention output [batch_size, tgt_len, d_model]

        Caching Strategy:
            First call (encoder_output provided):
                - Compute K and V projections from encoder_output
                - Store in cache if use_cache=True
                - Return cache dict

            Subsequent calls (encoder_output=None):
                - Retrieve K and V from cache
                - Use cached values for attention
                - No recomputation of encoder projections

        Efficiency:
            Without cache: O(src_len * tgt_len) per decoding step
            With cache: O(src_len) once + O(src_len) per step (loading from cache)
            Total savings over tgt_len steps: O(tgt_len * src_len) - O(src_len)
        """
        query = split_heads(query @ self.W_q, self.num_heads, self.d_k)
        if key is not None:
            key = split_heads(key @ self.W_k, self.num_heads, self.d_k)
            if use_cache:
                if self.cached_K is not None:
                    key = concat(self.cached_K, key, axis=-2)
                self.cached_K = key
        else:
            key = self.cached_K
        if value is not None:
            value = split_heads(value @ self.W_v, self.num_heads, self.d_k)
            if use_cache:
                if self.cached_V is not None:
                    value = concat(self.cached_V, value, axis=-2)
                self.cached_V = value
        else:
            value = self.cached_V
        output = self.attention(query, key, value, mask=mask)
        output = combine_heads(output, self.d_model)
        return output @ self.W_o

    def clear_cache(self):
        """Clear cached K/V projections."""
        self.cached_K = None
        self.cached_V = None
        self.encoder_output_cached = None


# ============================================================================
# Causal Masking for Autoregressive Models
# ============================================================================

class CausalMask(Module):
    """
    Generates and applies causal masks for autoregressive attention.

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    https://arxiv.org/abs/1706.03762

    Causal masks ensure each position can only attend to current and previous
    positions, preventing information leakage during training.
    """

    def __init__(self):
        """Initialize CausalMask module."""
        super().__init__()

    def extra_repr(self) -> str:
        """Extra representation for printing module details."""
        return "causal masking utilities"

    @staticmethod
    def create_causal_mask(seq_len: int, dtype=np.float32) -> np.ndarray:
        """
        Create a causal mask matrix.

        Args:
            seq_len (int): Sequence length
            dtype (np.dtype): Data type for mask (float32 or bool)

        Returns:
            np.ndarray: Causal mask of shape [seq_len, seq_len]

        Mask Value Semantics:
            For float dtype (used with additive masking):
                - 0.0: position can attend (not masked)
                - -1e9 or -inf: position cannot attend (masked out)

            For bool dtype (used with boolean indexing):
                - True: position can attend
                - False: position cannot attend

        Algorithm:
            1. Create lower triangular matrix
            2. For float dtype: return 0 where True, -inf where False
            3. For bool dtype: return the triangular matrix directly

        Example:
            seq_len = 3
            Output (float):
            [[0.0,  -inf,  -inf],
             [0.0,  0.0,   -inf],
             [0.0,  0.0,   0.0]]

            Or (bool):
            [[True, False, False],
             [True, True,  False],
             [True, True,  True]]
        """
        if dtype is np.float32:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-np.inf)
        elif dtype is bool or dtype is np.bool:
            mask = (1 - np.triu(np.ones((seq_len, seq_len)), k=1)).astype(np.bool)
        else:
            raise TypeError(f"dtype {dtype} is not supported")
        return mask

    @staticmethod
    def create_padding_mask(sequence_lengths: np.ndarray, max_seq_len: int,
                           dtype=np.float32) -> np.ndarray:
        """
        Create padding mask for variable-length sequences in a batch.

        Args:
            sequence_lengths (np.ndarray): Array of actual lengths [batch_size]
            max_seq_len (int): Maximum sequence length in batch
            dtype (np.dtype): Data type for mask

        Returns:
            np.ndarray: Padding mask of shape [batch_size, 1, max_seq_len]
                       Broadcasting-compatible with [batch_size, num_heads, seq_len_q, seq_len_k]

        Semantics:
            For float dtype:
                - 0.0: position is valid (not padded)
                - -1e9: position is padding (should be masked)

            For bool dtype:
                - True: position is valid
                - False: position is padding

        Algorithm:
            1. For each batch element b with actual length sequence_lengths[b]:
               - Positions 0 to sequence_lengths[b]-1: not masked
               - Positions sequence_lengths[b] to max_seq_len-1: masked

            2. Return shape [batch_size, 1, max_seq_len] for broadcasting

        Use Case:
            Combine with causal mask using element-wise operations:
            combined_mask = causal_mask + padding_mask
        """
        B = sequence_lengths.shape[0]
        # valid[b, j] = True if j < sequence_lengths[b]
        valid = np.arange(max_seq_len)[None, :] < sequence_lengths[:, None]
        if dtype is np.float32:
            mask = np.where(valid, 0.0, -np.inf).astype(np.float32)
        elif dtype is bool or dtype is np.bool:
            mask = valid
        else:
            raise TypeError(f"dtype {dtype} is not supported")
        return mask[:, np.newaxis, :]

    @staticmethod
    def create_causal_padding_mask(sequence_lengths: np.ndarray, max_seq_len: int,
                                   dtype=np.float32) -> np.ndarray:
        """
        Create combined causal and padding mask.

        Args:
            sequence_lengths (np.ndarray): Actual lengths [batch_size]
            max_seq_len (int): Maximum sequence length
            dtype (np.dtype): Data type for mask

        Returns:
            np.ndarray: Combined mask of shape [batch_size, max_seq_len, max_seq_len]

        Semantics:
            Position [b, i, j] is:
                - Unmasked (0.0 or True) if j <= i AND j < sequence_lengths[b]
                - Masked (-1e9 or False) otherwise

        Algorithm:
            1. Create causal mask [max_seq_len, max_seq_len]
            2. Create padding mask [batch_size, 1, max_seq_len]
            3. Combine: causal_mask + padding_mask[:,np.newaxis,:]
               (Broadcasting combines row-wise padding check)

        Shape Broadcasting:
            Causal mask: [max_seq_len, max_seq_len]
            Padding mask: [batch_size, 1, max_seq_len]
            Combined: [batch_size, max_seq_len, max_seq_len]
                      Each batch element gets causal mask + its padding
        """
        B = sequence_lengths.shape[0]
        valid = np.arange(max_seq_len)[None, :] < sequence_lengths[:, None]
        if dtype is np.float32:
            mask = np.ones((B, max_seq_len, max_seq_len))
            mask[np.arange(B), :sequence_lengths, :sequence_lengths] = 0.0
            mask *= -np.inf
        elif dtype is bool or dtype is np.bool:
            mask = np.ones((B, max_seq_len, max_seq_len))
            mask[np.arange(B), :sequence_lengths] = 0.0
            mask = 1 - mask
            mask = mask.astype(np.bool)
        else:
            raise TypeError(f"dtype {dtype} is not supported")
        return mask

    @staticmethod
    def create_sliding_window_mask(seq_len: int, window_size: int,
                                   dtype=np.float32) -> np.ndarray:
        """
        Create sliding window attention mask (recent tokens only).

        Args:
            seq_len (int): Sequence length
            window_size (int): Number of recent positions to attend to
            dtype (np.dtype): Data type for mask

        Returns:
            np.ndarray: Sliding window mask of shape [seq_len, seq_len]

        Mask Semantics:
            Position [i, j] is:
                - Unmasked (0.0 or True) if i - window_size < j <= i
                - Masked (-1e9 or False) otherwise

            Combines causal (j <= i) with recency (j > i - window_size)

        Example (seq_len=5, window_size=2):
            [[1, 0, 0, 0, 0],     Position 0: can attend to 0
             [1, 1, 0, 0, 0],     Position 1: can attend to 0-1
             [0, 1, 1, 0, 0],     Position 2: can attend to 1-2 (window)
             [0, 0, 1, 1, 0],     Position 3: can attend to 2-3 (window)
             [0, 0, 0, 1, 1]]     Position 4: can attend to 3-4 (window)

        Efficiency:
            Reduces attention computation from O(n^2) to O(n * window_size)
            Useful for long sequences where full attention is expensive

        Use Cases:
            - Long-context language models (e.g., LLaMA with long contexts)
            - Efficient transformer variants (Longformer, BigBird)
            - Time series with local temporal dependencies
        """
        i, j = np.ogrid[0:seq_len, 0:seq_len]
        mask_bool = (j <= i) & (j > i - window_size)
        if dtype is bool or dtype is np.bool:
            return mask_bool
        else:
            return np.where(mask_bool, 0.0, -1e9).astype(dtype)

    @staticmethod
    def create_local_attention_mask(seq_len: int, local_size: int, stride: Optional[int] = None,
                                    dtype=np.float32) -> np.ndarray:
        """
        Create local/sparse attention mask for efficiency.

        Args:
            seq_len (int): Sequence length
            local_size (int): Size of local attention window
            stride (int, optional): Stride between local windows. If None, stride = local_size
            dtype (np.dtype): Data type for mask

        Returns:
            np.ndarray: Local attention mask of shape [seq_len, seq_len]

        Concept:
            Divide sequence into overlapping windows and only attend within windows.
            Reduces attention from O(n^2) to O(n * local_size).

        Example (seq_len=8, local_size=3, stride=2):
            Windows:
            - Window 0: positions [0, 1, 2]
            - Window 1: positions [2, 3, 4]
            - Window 2: positions [4, 5, 6]
            - Window 3: positions [6, 7]

            Each position attends to positions within its local window.

        Use Cases:
            - Efficient transformers (Longformer, BigBird, Performer)
            - Bioinformatics: local structure importance (DNA, proteins)
            - Time series: local temporal patterns
        """
        raise NotImplementedError(
            "CausalMask.create_local_attention_mask() requires implementation.\n"
            "Hints:\n"
            "  1. Set stride to local_size if not provided\n"
            "  2. Create empty mask: mask = np.zeros((seq_len, seq_len), dtype=bool)\n"
            "  3. For each position i in range(seq_len):\n"
            "     - Determine which window(s) it belongs to\n"
            "     - Set mask[i, j] = True for j in local window around i\n"
            "  4. Convert to dtype (float: 0 or -1e9, bool: as is)\n"
        )


# ============================================================================
# Multi-Query Attention (MQA)
# ============================================================================

class MultiQueryAttention(Module):
    """
    Multi-Query Attention: shared key-value across query heads.

    Reference: "Fast Transformer Decoding: One Write-Head is All You Need"
    (Shazeer, 2019) https://arxiv.org/abs/1911.02727

    Unlike standard multi-head attention, uses single K and V projections
    shared across all query heads, reducing KV cache overhead during inference.

    Attributes:
        d_model (int): Total embedding dimension
        num_heads (int): Number of query heads
        d_k (int): Dimension of key/query per head
        d_v (int): Dimension of value per head
        dropout_p (float): Dropout probability on attention weights
        W_q_list (list): Query projections, one per head [d_model, d_k] each
        W_k (Parameter): Shared key projection [d_model, d_k]
        W_v (Parameter): Shared value projection [d_model, d_v]
        W_o (Parameter): Output projection [d_model, d_model]
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        """
        Initialize multi-query attention.

        Args:
            d_model (int): Embedding dimension
            num_heads (int): Number of query heads (KV heads = 1)
            dropout_p (float): Dropout probability. Default: 0.0
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.dropout_p = dropout_p

        # Query projection: single combined matrix [d_model, d_model]
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.W_q = Parameter(np.random.randn(d_model, d_model) * scale)

        # Shared key/value projections: [d_model, d_k]
        scale_kv = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_k = Parameter(np.random.randn(d_model, self.d_k) * scale_kv)
        self.W_v = Parameter(np.random.randn(d_model, self.d_v) * scale_kv)

        # Output projection
        self.W_o = Parameter(np.random.randn(d_model, d_model) * scale)

        # Attention object
        self.attention = ScaledDotProductAttention(dropout_p=dropout_p)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-query attention.

        Args:
            query (Tensor): Query of shape [batch_size, seq_len_q, d_model]
            key (Tensor): Key of shape [batch_size, seq_len_k, d_model]
            value (Tensor): Value of shape [batch_size, seq_len_v, d_model]
            mask (Tensor, optional): Attention mask

        Returns:
            output (Tensor): Output of shape [batch_size, seq_len_q, d_model]

        Algorithm:
            1. Project Q with combined matrix, split into heads:
               Q = split_heads(query @ W_q)  [B, num_heads, L_q, d_k]

            2. Project shared K and V (single head, broadcast):
               K = (key @ W_k)[:, None, :, :]    [B, 1, L_k, d_k]
               V = (value @ W_v)[:, None, :, :]  [B, 1, L_v, d_v]

            3. Apply attention (K/V broadcast across all Q heads):
               output = Attention(Q, K, V)  [B, num_heads, L_q, d_v]

            4. Combine heads and project:
               output = combine_heads(output) @ W_o  [B, L_q, d_model]
        """
        query = split_heads(query @ self.W_q, self.num_heads, self.d_k)
        key = (key @ self.W_k)[:, None, :, :]
        value = (value @ self.W_v)[:, None, :, :]
        output = self.attention(query, key, value, mask=mask)
        output = combine_heads(output, self.d_model)
        return output @ self.W_o

    def extra_repr(self) -> str:
        """Extra representation for printing module details."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, dropout_p={self.dropout_p}"



# ============================================================================
# Grouped-Query Attention (GQA)
# ============================================================================

class GroupedQueryAttention(Module):
    """
    Grouped-Query Attention: g groups of key-value heads.

    Reference: "GQA: Training Generalist Models for Diverse Question Answering"
    (Ainslie et al., 2023) https://arxiv.org/abs/2305.13245

    Shares key-value across multiple query heads within each group,
    providing a configurable trade-off between memory efficiency and quality.

    Attributes:
        d_model (int): Total embedding dimension
        num_heads (int): Number of query heads
        num_kv_groups (int): Number of KV groups (g)
        d_k (int): Dimension per head
        d_v (int): Dimension per head
        dropout_p (float): Dropout probability
        W_q_list (list): Query projections [d_model, d_k] per query head
        W_k_list (list): Key projections [d_model, d_k] per KV group
        W_v_list (list): Value projections [d_model, d_v] per KV group
        W_o (Parameter): Output projection [d_model, d_model]
    """

    def __init__(self, d_model: int, num_heads: int, num_kv_groups: int, dropout_p: float = 0.0):
        """
        Initialize grouped-query attention.

        Args:
            d_model (int): Embedding dimension
            num_heads (int): Number of query heads
            num_kv_groups (int): Number of KV groups (1 <= num_kv_groups <= num_heads)
            dropout_p (float): Dropout probability. Default: 0.0

        Raises:
            AssertionError: If num_heads not divisible by num_kv_groups
        """
        super().__init__()

        assert num_heads % num_kv_groups == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_groups ({num_kv_groups})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.dropout_p = dropout_p

        # Query projection: single combined matrix [d_model, d_model]
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.W_q = Parameter(np.random.randn(d_model, d_model) * scale)

        # Key/value projections: [d_model, num_kv_groups * d_k]
        kv_dim = num_kv_groups * self.d_k
        scale_kv = np.sqrt(2.0 / (d_model + kv_dim))
        self.W_k = Parameter(np.random.randn(d_model, kv_dim) * scale_kv)
        self.W_v = Parameter(np.random.randn(d_model, kv_dim) * scale_kv)

        # Output projection
        self.W_o = Parameter(np.random.randn(d_model, d_model) * scale)

        # Attention object
        self.attention = ScaledDotProductAttention(dropout_p=dropout_p)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of grouped-query attention.

        Args:
            query (Tensor): Query of shape [batch_size, seq_len_q, d_model]
            key (Tensor): Key of shape [batch_size, seq_len_k, d_model]
            value (Tensor): Value of shape [batch_size, seq_len_v, d_model]
            mask (Tensor, optional): Attention mask

        Returns:
            output (Tensor): Output of shape [batch_size, seq_len_q, d_model]

        Algorithm:
            1. Project Q with combined matrix, split into heads:
               Q = split_heads(query @ W_q)  [B, num_heads, L_q, d_k]

            2. Project K/V, split into KV groups:
               K = split_heads(key @ W_k, num_kv_groups)    [B, num_kv_groups, L_k, d_k]
               V = split_heads(value @ W_v, num_kv_groups)  [B, num_kv_groups, L_k, d_v]

            3. For each KV group g:
               - Slice Q heads for this group: Q_g = Q[:, g*gs:(g+1)*gs]  [B, gs, L_q, d_k]
               - K_g = K[:, g:g+1]  [B, 1, L_k, d_k]  (broadcasts across gs query heads)
               - head_output = Attention(Q_g, K_g, V_g)

            4. Concatenate all group outputs: [B, num_heads, L_q, d_v]
            5. Combine heads and project: output = combine_heads(output) @ W_o
        """
        query = split_heads(query @ self.W_q, self.num_heads, self.d_k)
        key = split_heads(key @ self.W_k, self.num_kv_groups, self.d_k)
        value = split_heads(value @ self.W_v, self.num_kv_groups, self.d_v)

        head_outputs = []
        for g in range(self.num_kv_groups):
            start = g * self.group_size
            q_g = query[:, start:start + self.group_size, :, :]
            k_g = key[:, g:g + 1, :, :]
            v_g = value[:, g:g + 1, :, :]
            out_g = self.attention(q_g, k_g, v_g, mask=mask)
            head_outputs.append(out_g)

        output = head_outputs[0]
        for i in range(1, len(head_outputs)):
            output = concat(output, head_outputs[i], axis=1)

        output = combine_heads(output, self.d_model)
        return output @ self.W_o

    def extra_repr(self) -> str:
        """Extra representation for printing module details."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, num_kv_groups={self.num_kv_groups}, dropout_p={self.dropout_p}"

    def convert_from_mha(self, mha_model, group_assignment: str = 'uniform'):
        """
        Initialize GQA from a trained Multi-Head Attention model.

        Args:
            mha_model: A trained MultiHeadAttention model with num_heads = self.num_heads
            group_assignment (str): How to initialize KV weights from MHA:
                - 'uniform': average KV weights within each group
                - 'first': use first head's weights per group
                - 'random': reinitialize KV weights randomly

        Returns:
            None (modifies self in place)

        Use Case:
            When converting a pre-trained MHA model to GQA for faster inference,
            initialize shared K/V weights intelligently.

        Implementation:
            For group j with query heads q*q_per_group to (j+1)*q_per_group:
            - W_k_j = mean(mha_model.W_k[q*q_per_group:(j+1)*q_per_group])
            - W_v_j = mean(mha_model.W_v[q*q_per_group:(j+1)*q_per_group])
        """
        raise NotImplementedError(
            "GroupedQueryAttention.convert_from_mha() requires implementation.\n"
            "Hints:\n"
            "  1. Verify mha_model.num_heads == self.num_heads\n"
            "  2. Copy query weights directly:\n"
            "     self.W_q_list = list(mha_model.W_q_list)  (same heads)\n"
            "  3. For each group j:\n"
            "     - Compute start_idx = j * q_per_group\n"
            "     - Get MHA K/V weights for heads in group\n"
            "     - If 'uniform': average them\n"
            "     - If 'first': take first head's weights\n"
            "     - If 'random': reinitialize\n"
            "  4. Copy output projection: self.W_o = mha_model.W_o\n"
        )

    def get_kv_cache_stats(self, max_seq_len: int) -> Dict:
        """
        Compare KV cache size with MHA and MQA.

        Args:
            max_seq_len (int): Maximum sequence length

        Returns:
            dict: Cache comparison statistics

        Analysis:
            GQA cache: num_kv_groups * [seq_len, d_k + d_v]
            MHA cache: num_heads * [seq_len, d_k + d_v]
            MQA cache: [seq_len, d_k + d_v]
            Savings vs MHA: num_heads / num_kv_groups
        """
        raise NotImplementedError(
            "GroupedQueryAttention.get_kv_cache_stats() requires implementation.\n"
            "Hints:\n"
            "  1. Compute cache sizes:\n"
            "     - gqa_bytes = max_seq_len * num_kv_groups * (d_k + d_v) * 4  (float32)\n"
            "     - mha_bytes = max_seq_len * num_heads * (d_k + d_v) * 4\n"
            "     - mqa_bytes = max_seq_len * (d_k + d_v) * 4\n"
            "  2. Compute savings factors\n"
            "  3. Return dict with all comparisons\n"
        )

