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

from .module import Module, Parameter
from ..foundations.functionals import Function


# ============================================================================
# Utility Functions
# ============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax implementation.

    Args:
        x (np.ndarray): Input array
        axis (int): Axis along which to normalize

    Returns:
        np.ndarray: Softmax output with sum of 1 along the specified axis

    Numerical Stability:
        Subtracts max value before exponentiation to prevent overflow:
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


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

    def extra_repr(self) -> str:
        """Extra representation for printing module details."""
        return f"dropout_p={self.dropout_p}"

    def forward(self, query, key, value, mask=None, training=False):
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
        raise NotImplementedError(
            "ScaledDotProductAttention.forward() requires implementation.\n"
            "Hints:\n"
            "  1. Extract d_k from query.shape[-1]\n"
            "  2. Compute scores = (query @ key.swapaxes(-2, -1)) / sqrt(d_k)\n"
            "  3. If mask is provided:\n"
            "     - If boolean mask: scores = np.where(mask, scores, -1e9)\n"
            "     - If float mask: scores = scores + mask (assumes -inf in mask)\n"
            "  4. Apply softmax along last dimension: weights = softmax(scores)\n"
            "  5. If training and dropout_p > 0: apply dropout to weights\n"
            "  6. Compute output = weights @ value\n"
            "  7. Return output and weights for visualization/analysis\n"
        )


# ============================================================================
# Multi-Head Attention
# ============================================================================

class MultiHeadAttention(Module):
    """
    Multi-Head Attention mechanism with learnable projections.

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

    def forward(self, query, key, value, mask=None, training=False):
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
        raise NotImplementedError(
            "MultiHeadAttention.forward() requires implementation.\n"
            "Hints:\n"
            "  1. Get batch_size and seq_len from query.shape[0:2]\n"
            "  2. Project: Q = query @ W_q, K = key @ W_k, V = value @ W_v\n"
            "  3. Reshape to separate heads:\n"
            "     Q = Q.reshape(batch_size, seq_len_q, num_heads, d_k)\n"
            "     K = K.reshape(batch_size, seq_len_k, num_heads, d_k)\n"
            "     V = V.reshape(batch_size, seq_len_v, num_heads, d_k)\n"
            "  4. Transpose to [batch, num_heads, seq_len, d_k]:\n"
            "     Q = Q.transpose(0, 2, 1, 3)  # [B, h, L_q, d_k]\n"
            "     K = K.transpose(0, 2, 1, 3)  # [B, h, L_k, d_k]\n"
            "     V = V.transpose(0, 2, 1, 3)  # [B, h, L_v, d_k]\n"
            "  5. Apply attention per head (loop or vectorized):\n"
            "     For each head i:\n"
            "       attn_out_i = self.attention.forward(Q[:, i], K[:, i], V[:, i], mask, training)\n"
            "  6. Concatenate heads: concat shape [B, L_q, d_model]\n"
            "  7. Output projection: output = concat @ W_o\n"
            "  8. Return output and dict of attention_weights per head\n"
        )

    def extra_repr(self) -> str:
        """Extra representation for printing module details."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, dropout_p={self.dropout_p}"


# ============================================================================
# Cross-Attention for Encoder-Decoder Models
# ============================================================================

class CrossAttention(MultiHeadAttention):
    """
    Cross-Attention module for encoder-decoder attention.

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    https://arxiv.org/abs/1706.03762

    Allows decoder to attend to encoder outputs. Inherits from MultiHeadAttention
    but with different query vs key/value sources.

    Attributes:
        Same as MultiHeadAttention, but projections handle different source spaces.
    """

    def forward(self, query, encoder_output, mask=None, training=False):
        """
        Forward pass of cross-attention.

        Args:
            query (Tensor): Decoder hidden state [batch_size, tgt_len, d_model]
            encoder_output (Tensor): Encoder output [batch_size, src_len, d_model]
            mask (Tensor, optional): Mask for encoder positions
                                    Shape: [batch_size, 1, src_len] for broadcasting
                                    or [batch_size, tgt_len, src_len]
                                    Default: None (no masking)
            training (bool): Whether in training mode

        Returns:
            output (Tensor): Cross-attention output [batch_size, tgt_len, d_model]
            attention_weights (dict): Attention weights for visualization

        Algorithm (same as MultiHeadAttention, just different inputs):
            1. Project query from decoder:
               Q_proj = query @ W_q  [B, L_t, d_model]

            2. Project key/value from encoder:
               K_proj = encoder_output @ W_k  [B, L_s, d_model]
               V_proj = encoder_output @ W_v  [B, L_s, d_model]

            3. Split heads and apply attention per head:
               For head i:
                   head_output_i = Attention(Q_proj_i, K_proj_i, V_proj_i)
                   with optional encoder padding mask

            4. Concatenate and project output:
               output = Concat(heads) @ W_o  [B, L_t, d_model]

        Note:
            No causal masking applied here. Decoder can attend to all encoder positions.
            If encoder has variable lengths, padding_mask should be provided.

        Shape Notes:
            - query (decoder): [B, tgt_len, d_model]
            - encoder_output: [B, src_len, d_model]
            - tgt_len != src_len in general
            - Attention matrix: [B, num_heads, tgt_len, src_len]
            - Output: [B, tgt_len, d_model]
        """
        raise NotImplementedError(
            "CrossAttention.forward() requires implementation.\n"
            "Hints:\n"
            "  Same as MultiHeadAttention.forward(), but:\n"
            "  - Query comes from decoder: query (first arg)\n"
            "  - Key and Value come from encoder: encoder_output (second arg)\n"
            "  - No causal masking (decoder can attend to all encoder positions)\n"
            "  - May have padding mask for variable-length sequences in encoder\n"
            "  Implementation steps:\n"
            "  1. Get shapes: batch_size, tgt_len = query.shape[0:2]\n"
            "  2. Project:\n"
            "     Q = query @ W_q  [B, tgt_len, d_model]\n"
            "     K = encoder_output @ W_k  [B, src_len, d_model]\n"
            "     V = encoder_output @ W_v  [B, src_len, d_model]\n"
            "  3. Split heads: Q -> [B, h, L_t, d_k], K -> [B, h, L_s, d_k], etc.\n"
            "  4. Apply attention per head with mask\n"
            "  5. Concatenate and project output\n"
        )

    def extra_repr(self) -> str:
        """Extra representation for printing module details."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, dropout_p={self.dropout_p}"


class CachedCrossAttention(CrossAttention):
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

    def forward(self, query, encoder_output=None, mask=None, training=False, use_cache=True):
        """
        Forward pass with optional K/V caching.

        Args:
            query (Tensor): Decoder query [batch_size, tgt_len, d_model]
            encoder_output (Tensor, optional): Encoder output [batch_size, src_len, d_model]
                                              Required on first call, can be None on subsequent calls
                                              if use_cache=True
            mask (Tensor, optional): Encoder padding mask
            training (bool): Whether in training mode
            use_cache (bool): Whether to use/create K/V cache. Default: True

        Returns:
            output (Tensor): Cross-attention output [batch_size, tgt_len, d_model]
            attention_weights (dict): Attention weights
            cache (dict): Cache dict with 'K' and 'V' if use_cache=True, else None

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
        raise NotImplementedError(
            "CachedCrossAttention.forward() requires implementation.\n"
            "Hints:\n"
            "  1. If encoder_output is provided:\n"
            "     - Project K/V from encoder: K = encoder @ W_k, V = encoder @ W_v\n"
            "     - Split heads for multi-head attention\n"
            "     - If use_cache: store self.cached_K and self.cached_V\n"
            "  2. If encoder_output is None:\n"
            "     - Retrieve from cache: K = self.cached_K, V = self.cached_V\n"
            "     - If cache is None, raise error\n"
            "  3. Project query and split heads\n"
            "  4. Apply attention using cached/computed K/V\n"
            "  5. Return output, attention_weights, and cache dict\n"
        )

    def clear_cache(self):
        """Clear cached K/V projections."""
        self.cached_K = None
        self.cached_V = None
        self.encoder_output_cached = None


class MultimodalCrossAttention(CrossAttention):
    """
    Cross-Attention for multimodal inputs (e.g., text to image, image to text).

    Extends cross-attention to handle inputs from different modalities
    (text, image, audio, etc.) with different projection spaces.

    Useful for:
    - Image captioning (attend from text to image features)
    - Visual question answering (attend from question to image)
    - Audio-visual learning (attend from audio to video or vice versa)
    """

    def __init__(self, d_model: int, num_heads: int, encoder_modality: str = 'vision',
                 decoder_modality: str = 'text', dropout_p: float = 0.0):
        """
        Initialize multimodal cross-attention.

        Args:
            d_model (int): Output embedding dimension
            num_heads (int): Number of attention heads
            encoder_modality (str): Type of encoder modality ('vision', 'audio', 'text', etc.)
            decoder_modality (str): Type of decoder modality
            dropout_p (float): Dropout probability
        """
        super().__init__(d_model, num_heads, dropout_p)
        self.encoder_modality = encoder_modality
        self.decoder_modality = decoder_modality

    def extra_repr(self) -> str:
        """Extra representation for printing module details."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, encoder_modality={self.encoder_modality}, decoder_modality={self.decoder_modality}"

    def forward(self, decoder_input, encoder_output, encoder_modality_type=None,
                mask=None, training=False):
        """
        Forward pass for multimodal cross-attention.

        Args:
            decoder_input (Tensor): Decoder tensor (typically text embeddings)
            encoder_output (Tensor): Encoder tensor (possibly different modality)
            encoder_modality_type (str, optional): Override stored modality type
            mask (Tensor, optional): Padding mask for encoder
            training (bool): Whether in training mode

        Returns:
            output (Tensor): Cross-modality attended output
            attention_weights (dict): Attention weights (useful for visualization)

        Notes:
            - Encoder/decoder may have different d_model dimensions
            - Projections adapt to modality-specific dimensions
            - Attention mechanism itself is modality-agnostic
        """
        raise NotImplementedError(
            "MultimodalCrossAttention.forward() requires implementation.\n"
            "Hints:\n"
            "  1. Similar to standard CrossAttention.forward()\n"
            "  2. May need modality-specific preprocessing:\n"
            "     - Vision: patch embeddings, pooling, or CNN features\n"
            "     - Audio: spectral processing or audio embeddings\n"
            "     - Text: standard token embeddings\n"
            "  3. Project decoder_input as query\n"
            "  4. Project encoder_output as key/value\n"
            "  5. Apply standard multi-head attention\n"
            "  6. May include modality-specific normalization or scaling\n"
        )


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
        raise NotImplementedError(
            "CausalMask.create_causal_mask() requires implementation.\n"
            "Hints:\n"
            "  1. Create indices: i, j = np.ogrid[0:seq_len, 0:seq_len]\n"
            "     This creates row and column index matrices\n"
            "  2. Create lower triangular: mask_bool = i >= j\n"
            "  3. If dtype is bool: return mask_bool\n"
            "  4. If dtype is float:\n"
            "     - Create float array: mask_float = np.zeros((seq_len, seq_len), dtype=dtype)\n"
            "     - Set masked positions: mask_float[~mask_bool] = -1e9\n"
            "     - Or use: mask_float = np.where(mask_bool, 0, -1e9).astype(dtype)\n"
            "     - Return mask_float\n"
        )

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
        raise NotImplementedError(
            "CausalMask.create_padding_mask() requires implementation.\n"
            "Hints:\n"
            "  1. Create position indices: positions = np.arange(max_seq_len)[np.newaxis, :]\n"
            "     Shape: [1, max_seq_len]\n"
            "  2. Create valid mask:\n"
            "     valid_mask = positions < sequence_lengths[:, np.newaxis]\n"
            "     Shape: [batch_size, max_seq_len]\n"
            "  3. Reshape for broadcasting: [batch_size, 1, max_seq_len]\n"
            "  4. Convert to dtype:\n"
            "     - If bool: return as is\n"
            "     - If float: np.where(valid_mask, 0, -1e9)\n"
        )

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
        raise NotImplementedError(
            "CausalMask.create_causal_padding_mask() requires implementation.\n"
            "Hints:\n"
            "  1. Create causal mask: causal = create_causal_mask(max_seq_len, dtype)\n"
            "     Shape: [max_seq_len, max_seq_len]\n"
            "  2. Create padding mask: padding = create_padding_mask(sequence_lengths, max_seq_len, dtype)\n"
            "     Shape: [batch_size, 1, max_seq_len]\n"
            "  3. Combine with broadcasting:\n"
            "     if dtype is float:\n"
            "       combined = causal[np.newaxis, :, :] + padding[:, np.newaxis, :]\n"
            "     else (bool):\n"
            "       combined = causal[np.newaxis, :, :] & padding[:, np.newaxis, :]\n"
            "  4. Reshape if needed: [batch_size, max_seq_len, max_seq_len]\n"
        )

    @staticmethod
    def apply_causal_mask(attention_scores: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply causal (or any other) mask to attention scores before softmax.

        Args:
            attention_scores (np.ndarray): Attention logits of shape
                                          [batch_size, num_heads, seq_len_q, seq_len_k]
            mask (np.ndarray, optional): Pre-computed mask. If None, creates causal mask.
                                        Expected shape for broadcasting:
                                        [1, 1, seq_len_q, seq_len_k] or [seq_len_q, seq_len_k]

        Returns:
            np.ndarray: Masked attention scores (ready for softmax)

        Algorithm:
            1. If no mask provided: create_causal_mask(seq_len_k)
            2. Apply mask with broadcasting:
               - If mask is float: masked_scores = scores + mask
               - If mask is bool: masked_scores = np.where(mask, scores, -1e9)
            3. Return masked scores

        Important:
            - Masking must happen BEFORE softmax
            - Masked positions should have -inf or very large negative values
            - Softmax of -inf naturally becomes 0
        """
        raise NotImplementedError(
            "CausalMask.apply_causal_mask() requires implementation.\n"
            "Hints:\n"
            "  1. Extract seq_len_k from attention_scores.shape[-1]\n"
            "  2. If mask is None:\n"
            "     mask = create_causal_mask(seq_len_k, dtype=attention_scores.dtype)\n"
            "  3. If mask is float:\n"
            "     return attention_scores + mask[np.newaxis, np.newaxis, :, :]\n"
            "     (Broadcasting: [B, h, L_q, L_k] + [L_q, L_k])\n"
            "  4. If mask is bool:\n"
            "     masked = np.where(mask[np.newaxis, np.newaxis, :, :],\n"
            "                       attention_scores, -1e9)\n"
            "     return masked\n"
        )

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
        raise NotImplementedError(
            "CausalMask.create_sliding_window_mask() requires implementation.\n"
            "Hints:\n"
            "  1. Create row/col indices: i, j = np.ogrid[0:seq_len, 0:seq_len]\n"
            "  2. Create mask: mask_bool = (j <= i) & (j > i - window_size)\n"
            "     (causal AND within window)\n"
            "  3. Convert to dtype:\n"
            "     - If bool: return mask_bool\n"
            "     - If float: np.where(mask_bool, 0, -1e9).astype(dtype)\n"
        )

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
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.dropout_p = dropout_p

        # Query projections: one per head
        scale_q = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_q_list = [
            Parameter(np.random.randn(d_model, self.d_k) * scale_q)
            for _ in range(num_heads)
        ]

        # Shared key/value projections
        scale_kv = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_k = Parameter(np.random.randn(d_model, self.d_k) * scale_kv)
        self.W_v = Parameter(np.random.randn(d_model, self.d_v) * scale_kv)

        # Output projection
        scale_o = np.sqrt(2.0 / (d_model + d_model))
        self.W_o = Parameter(np.random.randn(d_model, d_model) * scale_o)

        # Attention object
        self.attention = ScaledDotProductAttention(dropout_p=dropout_p)

    def forward(self, query, key, value, mask=None, training=False):
        """
        Forward pass of multi-query attention.

        Args:
            query (Tensor): Query of shape [batch_size, seq_len_q, d_model]
            key (Tensor): Key of shape [batch_size, seq_len_k, d_model]
            value (Tensor): Value of shape [batch_size, seq_len_v, d_model]
            mask (Tensor, optional): Attention mask
            training (bool): Whether in training mode

        Returns:
            output (Tensor): Output of shape [batch_size, seq_len_q, d_model]
            attention_weights (list): Attention weights for each query head

        Algorithm:
            1. Project shared key and value (computed once):
               K = key @ W_k  [B, L_k, d_k]
               V = value @ W_v  [B, L_v, d_v]

            2. For each query head i:
               Q_i = query @ W_q_i  [B, L_q, d_k]
               head_i = Attention(Q_i, K, V)  [B, L_q, d_v]

            3. Concatenate all query head outputs:
               concat = [head_0, ..., head_{h-1}]  [B, L_q, d_model]

            4. Output projection:
               output = concat @ W_o

        Key-Value Cache Efficiency:
            The shared K and V mean during inference, we only cache:
            - 1 key tensor: [seq_len, d_k]
            - 1 value tensor: [seq_len, d_v]
            Instead of h copies each. Savings are h x for memory and bandwidth.
        """
        raise NotImplementedError(
            "MultiQueryAttention.forward() requires implementation.\n"
            "Hints:\n"
            "  1. Project shared K and V once:\n"
            "     K = key @ W_k  [B, L_k, d_k]\n"
            "     V = value @ W_v  [B, L_v, d_v]\n"
            "  2. For each head i (loop or vectorize):\n"
            "     Q_i = query @ W_q_list[i]  [B, L_q, d_k]\n"
            "     head_output_i = self.attention.forward(Q_i, K, V, mask, training)\n"
            "  3. Concatenate head outputs along last dim: [B, L_q, num_heads * d_v]\n"
            "     concat shape should be [B, L_q, d_model]\n"
            "  4. Apply output projection: output = concat @ W_o\n"
            "  5. Return output and list of attention_weights from each head\n"
        )

    def extra_repr(self) -> str:
        """Extra representation for printing module details."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, dropout_p={self.dropout_p}"

    def get_kv_cache_size(self, max_seq_len: int) -> Dict:
        """
        Compute key-value cache size compared to multi-head attention.

        Args:
            max_seq_len (int): Maximum sequence length

        Returns:
            dict: Cache statistics including MQA vs MHA comparison

        Analysis:
            MQA cache: [seq_len, d_k + d_v] = seq_len * 2*d_k
            MHA cache: [h, seq_len, d_k + d_v] = seq_len * h * 2*d_k
            Savings: h-fold reduction in memory
        """
        raise NotImplementedError(
            "MultiQueryAttention.get_kv_cache_size() requires implementation.\n"
            "Hints:\n"
            "  1. MQA cache per sequence position: d_k + d_v (shared across heads)\n"
            "  2. Total MQA cache: max_seq_len * (d_k + d_v) bytes (float32)\n"
            "  3. Compare with MHA: max_seq_len * num_heads * (d_k + d_v) bytes\n"
            "  4. Return dict with:\n"
            "     - 'mqa_bytes': MQA cache size\n"
            "     - 'mha_bytes': equivalent MHA cache size\n"
            "     - 'speedup_factor': num_heads (theoretical bandwidth improvement)\n"
        )


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
        self.queries_per_group = num_heads // num_kv_groups
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.dropout_p = dropout_p

        # Query projections: one per query head
        scale_q = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_q_list = [
            Parameter(np.random.randn(d_model, self.d_k) * scale_q)
            for _ in range(num_heads)
        ]

        # Key/value projections: one per group (shared within group)
        scale_kv = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_k_list = [
            Parameter(np.random.randn(d_model, self.d_k) * scale_kv)
            for _ in range(num_kv_groups)
        ]
        self.W_v_list = [
            Parameter(np.random.randn(d_model, self.d_v) * scale_kv)
            for _ in range(num_kv_groups)
        ]

        # Output projection
        scale_o = np.sqrt(2.0 / (d_model + d_model))
        self.W_o = Parameter(np.random.randn(d_model, d_model) * scale_o)

        # Attention object
        self.attention = ScaledDotProductAttention(dropout_p=dropout_p)

    def forward(self, query, key, value, mask=None, training=False):
        """
        Forward pass of grouped-query attention.

        Args:
            query (Tensor): Query of shape [batch_size, seq_len_q, d_model]
            key (Tensor): Key of shape [batch_size, seq_len_k, d_model]
            value (Tensor): Value of shape [batch_size, seq_len_v, d_model]
            mask (Tensor, optional): Attention mask
            training (bool): Whether in training mode

        Returns:
            output (Tensor): Output of shape [batch_size, seq_len_q, d_model]
            attention_weights (dict): Attention weights for each group

        Algorithm:
            1. Project all query heads:
               Q_i = query @ W_q_i for i in [0, num_heads)

            2. Project all KV groups:
               K_j = key @ W_k_j for j in [0, num_kv_groups)
               V_j = value @ W_v_j for j in [0, num_kv_groups)

            3. For each KV group j:
               - Get indices of query heads in this group
               - Collect corresponding Q projections: Q_{j,0}, ..., Q_{j,q_per_group-1}
               - For each query head in group:
                   head_output = Attention(Q_i, K_j, V_j)
               - Concatenate within group

            4. Concatenate all group outputs: [B, L_q, d_model]

            5. Apply output projection: output = concat @ W_o

        Group Assignment:
            Query heads are distributed to groups:
            - Group 0: query heads 0 to (q_per_group - 1)
            - Group 1: query heads q_per_group to (2 * q_per_group - 1)
            - Group j: query heads j*q_per_group to (j+1)*q_per_group - 1

            All query heads in group j share K_j and V_j.
        """
        raise NotImplementedError(
            "GroupedQueryAttention.forward() requires implementation.\n"
            "Hints:\n"
            "  1. Project all query heads:\n"
            "     Q_list = [query @ W_q_i for i in range(num_heads)]\n"
            "  2. Project all KV groups:\n"
            "     K_list = [key @ W_k_j for j in range(num_kv_groups)]\n"
            "     V_list = [value @ W_v_j for j in range(num_kv_groups)]\n"
            "  3. For each group j:\n"
            "     - Get query head indices: start_idx = j * q_per_group\n"
            "     - Collect Q heads in group: [Q_list[i] for i in range(start_idx, start_idx + q_per_group)]\n"
            "     - For each Q in group, compute: head_out = attention(Q, K_list[j], V_list[j])\n"
            "     - Concatenate head outputs within group\n"
            "  4. Concatenate all groups: [B, L_q, d_model]\n"
            "  5. Apply output projection: output = concat @ W_o\n"
            "  6. Return output and attention_weights dict\n"
        )

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


# ============================================================================
# Functional Operations (Stateful Functions for Autograd)
# ============================================================================

class ScaledDotProductAttentionFunction(Function):
    """
    Scaled Dot-Product Attention functional operation.

    The fundamental attention mechanism used in transformers.

    Math:
        scores = Q @ K^T / √d_k
        attention_weights = softmax(scores + mask)  # if mask provided
        output = attention_weights @ V

    Input shapes:
        Q: (batch, num_heads, seq_len_q, d_k)
        K: (batch, num_heads, seq_len_k, d_k)
        V: (batch, num_heads, seq_len_k, d_v)
        mask: (batch, 1, seq_len_q, seq_len_k) or broadcastable

    Output:
        (batch, num_heads, seq_len_q, d_v)
    """

    def __init__(self, dropout: float = 0.0):
        """
        Initialize ScaledDotProductAttention function.

        Args:
            dropout: Dropout probability applied to attention weights
        """
        self.dropout = dropout

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> np.ndarray:
        """
        Compute scaled dot-product attention.

        Args:
            Q: Query tensor (batch, heads, seq_q, d_k)
            K: Key tensor (batch, heads, seq_k, d_k)
            V: Value tensor (batch, heads, seq_k, d_v)
            mask: Attention mask (optional). Use -inf for positions to mask.
            training: Whether in training mode (for dropout)

        Returns:
            Attention output (batch, heads, seq_q, d_v)
        """
        raise NotImplementedError(
            "TODO: Implement ScaledDotProductAttention forward\n"
            "Hint:\n"
            "  # Store for backward\n"
            "  self.Q = Q\n"
            "  self.K = K\n"
            "  self.V = V\n"
            "  self.training = training\n"
            "  \n"
            "  d_k = Q.shape[-1]\n"
            "  \n"
            "  # Compute attention scores: Q @ K^T / sqrt(d_k)\n"
            "  # Shape: (batch, heads, seq_q, seq_k)\n"
            "  self.scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)\n"
            "  \n"
            "  # Apply mask (if provided)\n"
            "  if mask is not None:\n"
            "      self.scores = self.scores + mask  # mask should have -inf where masked\n"
            "  \n"
            "  # Softmax over keys\n"
            "  self.attn_weights = softmax(self.scores, axis=-1)\n"
            "  \n"
            "  # Apply dropout\n"
            "  if training and self.dropout > 0:\n"
            "      self.dropout_mask = (np.random.rand(*self.attn_weights.shape) > self.dropout).astype(np.float32)\n"
            "      self.attn_weights = self.attn_weights * self.dropout_mask / (1 - self.dropout)\n"
            "  \n"
            "  # Compute output: attention_weights @ V\n"
            "  output = self.attn_weights @ V\n"
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
            "  if self.training and self.dropout > 0:\n"
            "      grad_attn = grad_attn * self.dropout_mask / (1 - self.dropout)\n"
            "  \n"
            "  # Gradient through softmax\n"
            "  # For softmax: d_softmax = softmax * (grad - sum(grad * softmax))\n"
            "  sum_term = np.sum(grad_attn * self.attn_weights, axis=-1, keepdims=True)\n"
            "  grad_scores = self.attn_weights * (grad_attn - sum_term)\n"
            "  \n"
            "  # Scale gradient\n"
            "  grad_scores = grad_scores / np.sqrt(d_k)\n"
            "  \n"
            "  # Gradient w.r.t. Q and K\n"
            "  # scores = Q @ K^T\n"
            "  grad_Q = grad_scores @ self.K\n"
            "  grad_K = grad_scores.transpose(0, 1, 3, 2) @ self.Q\n"
            "  \n"
            "  return grad_Q, grad_K, grad_V"
        )


class MultiHeadAttentionFunction(Function):
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

    def __init__(self, num_heads: int, dropout: float = 0.0):
        """
        Initialize MultiHeadAttention function.

        Args:
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
        """
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_fn = ScaledDotProductAttentionFunction(dropout=dropout)

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
        W_O: np.ndarray,
        mask: Optional[np.ndarray] = None,
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
            mask: Attention mask (optional)
            training: Whether in training mode

        Returns:
            Output tensor (batch, seq_q, d_model)
        """
        raise NotImplementedError(
            "TODO: Implement MultiHeadAttention forward\n"
            "Hint:\n"
            "  # Store for backward\n"
            "  self.Q = Q\n"
            "  self.K = K\n"
            "  self.V = V\n"
            "  self.W_Q = W_Q\n"
            "  self.W_K = W_K\n"
            "  self.W_V = W_V\n"
            "  self.W_O = W_O\n"
            "  \n"
            "  batch_size, seq_len_q, d_model = Q.shape\n"
            "  _, seq_len_k, _ = K.shape\n"
            "  d_k = d_model // self.num_heads\n"
            "  \n"
            "  # Project Q, K, V\n"
            "  Q_proj = Q @ W_Q  # (batch, seq_q, d_model)\n"
            "  K_proj = K @ W_K  # (batch, seq_k, d_model)\n"
            "  V_proj = V @ W_V  # (batch, seq_k, d_model)\n"
            "  \n"
            "  # Reshape to (batch, num_heads, seq, d_k)\n"
            "  Q_heads = Q_proj.reshape(batch_size, seq_len_q, self.num_heads, d_k).transpose(0, 2, 1, 3)\n"
            "  K_heads = K_proj.reshape(batch_size, seq_len_k, self.num_heads, d_k).transpose(0, 2, 1, 3)\n"
            "  V_heads = V_proj.reshape(batch_size, seq_len_k, self.num_heads, d_k).transpose(0, 2, 1, 3)\n"
            "  \n"
            "  # Store projected values\n"
            "  self.Q_heads = Q_heads\n"
            "  self.K_heads = K_heads\n"
            "  self.V_heads = V_heads\n"
            "  \n"
            "  # Apply scaled dot-product attention\n"
            "  attn_output = self.attention_fn.forward(Q_heads, K_heads, V_heads, mask, training)\n"
            "  self.attn_output = attn_output  # (batch, num_heads, seq_q, d_k)\n"
            "  \n"
            "  # Reshape back: (batch, seq_q, d_model)\n"
            "  concat_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, d_model)\n"
            "  self.concat_output = concat_output\n"
            "  \n"
            "  # Final projection\n"
            "  output = concat_output @ W_O\n"
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
            "  grad_W_O = self.concat_output.transpose(0, 2, 1).reshape(-1, d_model).T @ \\\n"
            "             grad_output.reshape(-1, d_model)\n"
            "  \n"
            "  # Gradient w.r.t. concat_output\n"
            "  grad_concat = grad_output @ self.W_O.T\n"
            "  \n"
            "  # Reshape to attention output shape\n"
            "  grad_attn_output = grad_concat.reshape(batch_size, seq_len_q, self.num_heads, d_k).transpose(0, 2, 1, 3)\n"
            "  \n"
            "  # Gradient through attention\n"
            "  grad_Q_heads, grad_K_heads, grad_V_heads = self.attention_fn.backward(grad_attn_output)\n"
            "  \n"
            "  # Reshape heads back\n"
            "  grad_Q_proj = grad_Q_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, d_model)\n"
            "  grad_K_proj = grad_K_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_k, d_model)\n"
            "  grad_V_proj = grad_V_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_k, d_model)\n"
            "  \n"
            "  # Gradient w.r.t. projection weights\n"
            "  grad_W_Q = self.Q.transpose(0, 2, 1).reshape(-1, d_model).T @ grad_Q_proj.reshape(-1, d_model)\n"
            "  grad_W_K = self.K.transpose(0, 2, 1).reshape(-1, d_model).T @ grad_K_proj.reshape(-1, d_model)\n"
            "  grad_W_V = self.V.transpose(0, 2, 1).reshape(-1, d_model).T @ grad_V_proj.reshape(-1, d_model)\n"
            "  \n"
            "  # Gradient w.r.t. inputs\n"
            "  grad_Q = grad_Q_proj @ self.W_Q.T\n"
            "  grad_K = grad_K_proj @ self.W_K.T\n"
            "  grad_V = grad_V_proj @ self.W_V.T\n"
            "  \n"
            "  return grad_Q, grad_K, grad_V, grad_W_Q, grad_W_K, grad_W_V, grad_W_O"
        )


class CrossAttentionFunction(Function):
    """
    Cross-Attention functional operation.

    Attention where queries come from one sequence and keys/values from another.
    Used in encoder-decoder architectures (e.g., original Transformer decoder).

    Input shapes:
        Q: (batch, seq_len_q, d_model) - from decoder
        K, V: (batch, seq_len_k, d_model) - from encoder
    """

    def __init__(self, num_heads: int, dropout: float = 0.0):
        """
        Initialize CrossAttention function.

        Args:
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        self.mha = MultiHeadAttentionFunction(num_heads=num_heads, dropout=dropout)

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
        W_O: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> np.ndarray:
        """
        Compute cross-attention.

        Args:
            Q: Query tensor from decoder (batch, seq_q, d_model)
            K: Key tensor from encoder (batch, seq_k, d_model)
            V: Value tensor from encoder (batch, seq_k, d_model)
            W_Q, W_K, W_V, W_O: Projection weights
            mask: Optional mask for encoder padding
            training: Whether in training mode

        Returns:
            Output tensor (batch, seq_q, d_model)
        """
        return self.mha.forward(Q, K, V, W_Q, W_K, W_V, W_O, mask, training)

    def backward(
        self,
        grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for cross-attention.

        Returns:
            Tuple of gradients
        """
        return self.mha.backward(grad_output)


class CausalSelfAttentionFunction(Function):
    """
    Causal (Masked) Self-Attention functional operation.

    Self-attention with causal masking to prevent attending to future positions.
    Used in autoregressive models like GPT.

    The mask ensures position i can only attend to positions 0...i.
    """

    def __init__(self, num_heads: int, dropout: float = 0.0):
        """
        Initialize CausalSelfAttention function.

        Args:
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        self.mha = MultiHeadAttentionFunction(num_heads=num_heads, dropout=dropout)

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
        W_O: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Compute causal self-attention.

        Args:
            Q, K, V: Input tensors (usually the same for self-attention)
            W_Q, W_K, W_V, W_O: Projection weights
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
            "  causal_mask = np.triu(np.ones((seq_len, seq_len)) * (-np.inf), k=1)\n"
            "  causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq, seq)\n"
            "  \n"
            "  return self.mha.forward(Q, K, V, W_Q, W_K, W_V, W_O, causal_mask, training)"
        )

    def backward(
        self,
        grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for causal self-attention.

        Returns:
            Tuple of gradients
        """
        return self.mha.backward(grad_output)
