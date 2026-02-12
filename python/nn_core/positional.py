"""
Positional Encoding Modules
===========================

This module provides positional encoding methods for transformer models.
Position information is crucial since self-attention is permutation-invariant.

Module Classes:
    - SinusoidalPositionalEncoding: Original Transformer sinusoidal encoding
    - LearnedPositionalEmbedding: Learned embeddings (BERT-style)
    - RelativePositionalEmbedding: Relative position biases (Transformer-XL)
    - RotaryPositionalEmbedding: RoPE for modern LLMs
    - ALiBiPositionalBias: Attention with Linear Biases

Helper Functions:
    - create_sinusoidal_encoding: Create sinusoidal encoding matrix
    - create_rope_encoding: Create RoPE cos/sin tables
    - compare_positional_encodings: Compare different methods

Theory Notes:
=============

SINUSOIDAL ENCODING (Transformer, 2017):
- Uses sin/cos at different frequencies
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
- Can extrapolate to longer sequences than training

LEARNED EMBEDDINGS (BERT, 2018):
- Position lookup table learned during training
- More expressive but cannot extrapolate
- Requires seq_length × d_model parameters

ROTARY POSITION EMBEDDING (RoPE, 2021):
- Applies rotation matrices to encode position
- Naturally captures relative positions in attention
- Standard for modern LLMs (GPT-3.5, Claude, etc.)
- Excellent extrapolation via interpolation

ALIBI (2022):
- Adds linear biases to attention scores
- bias(i, j) = -m × |i - j|
- Very simple yet effective for long context
- Minimal parameters (one slope per head)

REFERENCES:
- "Attention Is All You Need" Vaswani et al., 2017 - Sinusoidal
- "BERT" Devlin et al., 2018 - Learned embeddings
- "RoFormer" Su et al., 2021 - RoPE
- "Train Short, Test Long" Press et al., 2022 - ALiBi
"""

import math
import numpy as np
from typing import Tuple, Optional

from .module import Module, Parameter


# =============================================================================
# Sinusoidal Positional Encoding
# =============================================================================

class SinusoidalPositionalEncoding(Module):
    """
    Sinusoidal positional encoding from the original Transformer paper.

    This encoding is deterministic and can extrapolate to longer sequences
    than those seen during training.

    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Example:
        >>> pos_encoding = SinusoidalPositionalEncoding(d_model=512)
        >>> x = np.random.randn(batch_size, seq_length, 512)
        >>> encoded = x + pos_encoding.get_encoding(seq_length)

    Attributes:
        d_model (int): Embedding dimension
        max_seq_length (int): Maximum sequence length to pre-compute
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000):
        """
        Initialize sinusoidal positional encoding.

        Args:
            d_model: Embedding dimension (typically 512, 768, etc.)
            max_seq_length: Maximum sequence length to pre-compute encodings for

        Raises:
            ValueError: If d_model is not positive
        """
        raise NotImplementedError(
            "TODO: Initialize sinusoidal positional encoding\n"
            "1. Call super().__init__()\n"
            "2. Validate d_model > 0\n"
            "3. Store d_model and max_seq_length\n"
            "4. Create PE matrix of shape (max_seq_length, d_model):\n"
            "   - position = np.arange(max_seq_length)[:, np.newaxis]\n"
            "   - div_term = np.exp(np.arange(0, d_model, 2) * -(log(10000) / d_model))\n"
            "   - pe[:, 0::2] = sin(position * div_term)\n"
            "   - pe[:, 1::2] = cos(position * div_term)\n"
            "5. Register pe as buffer using self.register_buffer('pe', pe)"
        )

    def get_encoding(self, seq_length: int) -> np.ndarray:
        """
        Get positional encoding for a given sequence length.

        Args:
            seq_length: Length of sequence to encode

        Returns:
            Array of shape (seq_length, d_model) with positional encodings

        Raises:
            ValueError: If seq_length > max_seq_length
        """
        raise NotImplementedError(
            "TODO: Return positional encoding slice\n"
            "1. Validate seq_length <= max_seq_length\n"
            "2. Return self.pe[:seq_length]"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to embeddings.

        Args:
            x: Array of shape (batch_size, seq_length, d_model)

        Returns:
            Array of shape (batch_size, seq_length, d_model) with positional info added

        Raises:
            ValueError: If x.shape[-1] != d_model
            ValueError: If seq_length > max_seq_length
        """
        raise NotImplementedError(
            "TODO: Add positional encoding to input\n"
            "1. Validate x.shape[-1] == d_model\n"
            "2. Get seq_length from x.shape[1]\n"
            "3. Get encoding via get_encoding(seq_length)\n"
            "4. Return x + pe[np.newaxis, :, :]  # broadcast over batch"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"d_model={self.d_model}, max_seq_length={self.max_seq_length}"

    @staticmethod
    def compute_pe(d_model: int, seq_length: int) -> np.ndarray:
        """
        Static method to compute positional encoding matrix.

        Args:
            d_model: Embedding dimension
            seq_length: Sequence length

        Returns:
            Array of shape (seq_length, d_model)

        Formula:
            PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        raise NotImplementedError(
            "TODO: Compute sinusoidal positional encoding\n"
            "1. Create pe = np.zeros((seq_length, d_model))\n"
            "2. position = np.arange(seq_length)[:, np.newaxis]\n"
            "3. div_term = np.exp(np.arange(0, d_model, 2) * -(log(10000) / d_model))\n"
            "4. pe[:, 0::2] = np.sin(position * div_term)\n"
            "5. pe[:, 1::2] = np.cos(position * div_term)\n"
            "   (handle odd d_model: pe[:, 1::2] = cos(position * div_term[:-1]))\n"
            "6. Return pe"
        )


def create_sinusoidal_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Convenience function to create a sinusoidal encoding matrix.

    Args:
        seq_length: Sequence length
        d_model: Embedding dimension

    Returns:
        Array of shape (seq_length, d_model)

    Example:
        >>> pe = create_sinusoidal_encoding(100, 512)
        >>> assert pe.shape == (100, 512)
        >>> assert np.abs(pe).max() <= 1.0  # bounded
    """
    return SinusoidalPositionalEncoding.compute_pe(d_model, seq_length)


# =============================================================================
# Learned Positional Embeddings
# =============================================================================

class LearnedPositionalEmbedding(Module):
    """
    Learned positional embeddings as in BERT.

    Positions are treated as a lookup table that is learned during training.

    Example:
        >>> pos_embed = LearnedPositionalEmbedding(seq_length=512, d_model=768)
        >>> x = np.random.randn(batch_size, 256, 768)  # 256 tokens
        >>> encoded = x + pos_embed(np.arange(256))

    Attributes:
        seq_length (int): Maximum sequence length
        d_model (int): Embedding dimension
        pe (Parameter): Learnable embedding table
    """

    def __init__(
        self,
        seq_length: int,
        d_model: int,
        padding_idx: Optional[int] = None,
        initialization: str = "normal",
    ):
        """
        Initialize learned positional embeddings.

        Args:
            seq_length: Maximum sequence length (vocab_size for embedding)
            d_model: Embedding dimension
            padding_idx: Optional padding index (treated as zero vector)
            initialization: Initialization scheme - 'normal', 'uniform', 'xavier'

        Raises:
            ValueError: If seq_length or d_model are not positive
            ValueError: If padding_idx is out of range
        """
        raise NotImplementedError(
            "TODO: Initialize learned positional embeddings\n"
            "1. Call super().__init__()\n"
            "2. Validate seq_length > 0, d_model > 0\n"
            "3. Validate padding_idx in range if provided\n"
            "4. Store attributes\n"
            "5. Initialize weights based on initialization scheme:\n"
            "   - 'normal': np.random.normal(0, 0.02, (seq_length, d_model))\n"
            "   - 'uniform': np.random.uniform(-0.02, 0.02, ...)\n"
            "   - 'xavier': limit = sqrt(6/(seq_length+d_model)), uniform(-limit, limit)\n"
            "6. Create self.pe = Parameter(weight)\n"
            "7. Zero out padding_idx if specified"
        )

    def forward(self, position_ids: np.ndarray) -> np.ndarray:
        """
        Get positional embeddings for given position indices.

        Args:
            position_ids: Array of shape (...,) with position indices
                         Values should be in [0, seq_length)

        Returns:
            Array of shape (..., d_model) with positional embeddings

        Raises:
            ValueError: If any position_id >= seq_length
            ValueError: If any position_id < 0

        Example:
            >>> pos_ids = np.array([0, 1, 2, 3, 4])  # shape (5,)
            >>> embeddings = pos_embed(pos_ids)  # shape (5, 768)
        """
        raise NotImplementedError(
            "TODO: Look up positional embeddings\n"
            "1. Validate position_ids >= 0\n"
            "2. Validate position_ids < seq_length\n"
            "3. Return self.pe.data[position_ids]"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"seq_length={self.seq_length}, d_model={self.d_model}"


# =============================================================================
# Relative Positional Embeddings
# =============================================================================

class RelativePositionalEmbedding(Module):
    """
    Relative position embeddings for self-attention.

    Instead of absolute positions, uses relative position biases between
    query and key positions (Transformer-XL, DeBERTa style).

    Attention(Q, K, V) = softmax((QK^T + Relative_Position_Bias) / sqrt(d_k)) V

    Attributes:
        num_buckets (int): Number of relative position buckets
        d_model (int): Embedding dimension
        bidirectional (bool): Whether to use bidirectional relative positions
    """

    def __init__(
        self,
        num_buckets: int,
        d_model: int,
        max_distance: int = 128,
        bidirectional: bool = True,
    ):
        """
        Initialize relative positional embeddings.

        Args:
            num_buckets: Number of relative position buckets (typically 32)
            d_model: Embedding dimension
            max_distance: Maximum distance to bucket
            bidirectional: If True, use separate embeddings for ±relative_pos

        Raises:
            ValueError: If num_buckets or d_model not positive
        """
        raise NotImplementedError(
            "TODO: Initialize relative positional embeddings\n"
            "1. Call super().__init__()\n"
            "2. Validate num_buckets > 0, d_model > 0\n"
            "3. Store attributes\n"
            "4. vocab_size = 2*num_buckets if bidirectional else num_buckets\n"
            "5. Initialize embedding weights with normal(0, d_model^-0.5)\n"
            "6. Create self.embedding = Parameter(weight)"
        )

    def forward(self, seq_length: int) -> np.ndarray:
        """
        Compute relative position embeddings matrix.

        Args:
            seq_length: Length of sequence

        Returns:
            Array of shape (seq_length, seq_length, d_model)
            result[i, j] = embedding of relative position (j - i)

        Example:
            >>> rel_pos = RelativePositionalEmbedding(num_buckets=32, d_model=64)
            >>> pos_matrix = rel_pos(seq_length=128)
            >>> assert pos_matrix.shape == (128, 128, 64)
        """
        raise NotImplementedError(
            "TODO: Compute relative position embeddings\n"
            "1. query_pos = np.arange(seq_length)[:, None]\n"
            "2. key_pos = np.arange(seq_length)[None, :]\n"
            "3. relative_pos = key_pos - query_pos\n"
            "4. bucket_ids = _get_position_buckets(relative_pos, ...)\n"
            "5. Return self.embedding.data[bucket_ids]"
        )

    @staticmethod
    def _get_position_buckets(
        relative_pos: np.ndarray,
        num_buckets: int,
        max_distance: int,
        bidirectional: bool = True,
    ) -> np.ndarray:
        """
        Map relative positions to bucket indices.

        Args:
            relative_pos: Relative positions (can be negative)
            num_buckets: Number of buckets
            max_distance: Distances beyond this use max bucket
            bidirectional: Whether to handle negative positions separately

        Returns:
            Array of bucket indices (same shape as relative_pos)
        """
        raise NotImplementedError(
            "TODO: Map relative positions to bucket indices\n"
            "1. Create buckets array of zeros\n"
            "2. If bidirectional:\n"
            "   - Handle negative positions: bucket = num_buckets//2 - 1 - dist\n"
            "   - Handle positive positions: bucket = num_buckets//2 + dist\n"
            "3. If not bidirectional:\n"
            "   - bucket = min(abs(relative_pos), max_distance-1)\n"
            "4. Clip buckets to valid range\n"
            "5. Return buckets"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"num_buckets={self.num_buckets}, d_model={self.d_model}"


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================

class RotaryPositionalEmbedding(Module):
    """
    Rotary Position Embedding (RoPE) for modern LLMs.

    RoPE applies rotation matrices to encode position. Key insight:
    dot product of rotated vectors naturally encodes relative position.

    Example:
        >>> rope = RotaryPositionalEmbedding(d_model=128)
        >>> q = np.random.randn(batch_size, seq_len, 128)
        >>> k = np.random.randn(batch_size, seq_len, 128)
        >>> q_rotated, k_rotated = rope(q, k, seq_len=seq_len)

    Attributes:
        d_model (int): Embedding dimension (must be even)
        base (float): Frequency base (10000.0 is standard)
        max_seq_length (int): Max sequence length to pre-compute
    """

    def __init__(
        self,
        d_model: int,
        base: float = 10000.0,
        max_seq_length: int = 8192,
    ):
        """
        Initialize RoPE module.

        Args:
            d_model: Embedding dimension (must be even)
            base: Frequency base (10000.0 is standard)
            max_seq_length: Maximum sequence length to cache rotations

        Raises:
            ValueError: If d_model is odd
            ValueError: If base <= 0
        """
        raise NotImplementedError(
            "TODO: Initialize RoPE\n"
            "1. Call super().__init__()\n"
            "2. Validate d_model is even\n"
            "3. Validate base > 0\n"
            "4. Store attributes\n"
            "5. Compute inverse frequencies:\n"
            "   inv_freq = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))\n"
            "6. Pre-compute cos/sin tables for efficiency:\n"
            "   t = np.arange(max_seq_length)\n"
            "   freqs = np.outer(t, inv_freq)  # (seq_len, d_model/2)\n"
            "   Register cos_cached = np.cos(freqs), sin_cached = np.sin(freqs)"
        )

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        seq_length: int,
        offset: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply rotary embeddings to query and key.

        Args:
            q: Query array of shape (..., seq_length, d_model)
            k: Key array of shape (..., seq_length, d_model)
            seq_length: Sequence length
            offset: Position offset for batched inference

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes
        """
        raise NotImplementedError(
            "TODO: Apply RoPE to query and key\n"
            "1. Get cos/sin for positions [offset : offset+seq_length]\n"
            "2. Reshape q and k to pair adjacent dimensions:\n"
            "   q_reshaped = q.reshape(..., seq_length, d_model//2, 2)\n"
            "3. Apply rotation:\n"
            "   q_rotated[..., 0] = q[..., 0] * cos - q[..., 1] * sin\n"
            "   q_rotated[..., 1] = q[..., 0] * sin + q[..., 1] * cos\n"
            "4. Reshape back to original shape\n"
            "5. Return (q_rotated, k_rotated)"
        )

    @staticmethod
    def rotate_half(x: np.ndarray) -> np.ndarray:
        """
        Rotate half the hidden dims of x.

        Splits x into two halves and rotates:
        [x1, x2] -> [-x2, x1]
        """
        raise NotImplementedError(
            "TODO: Rotate half dimensions\n"
            "x1, x2 = x[..., :d//2], x[..., d//2:]\n"
            "return np.concatenate([-x2, x1], axis=-1)"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"d_model={self.d_model}, base={self.base}"


def create_rope_encoding(
    seq_length: int,
    d_model: int,
    base: float = 10000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create RoPE cos/sin tables.

    Args:
        seq_length: Sequence length
        d_model: Embedding dimension
        base: Frequency base

    Returns:
        Tuple of (cos_table, sin_table), each of shape (seq_length, d_model//2)
    """
    raise NotImplementedError(
        "TODO: Create RoPE encoding tables\n"
        "1. inv_freq = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))\n"
        "2. t = np.arange(seq_length)\n"
        "3. freqs = np.outer(t, inv_freq)\n"
        "4. Return (np.cos(freqs), np.sin(freqs))"
    )


# =============================================================================
# ALiBi (Attention with Linear Biases)
# =============================================================================

class ALiBiPositionalBias(Module):
    """
    Attention with Linear Biases (ALiBi).

    Adds position-dependent linear biases to attention scores:
    bias(i, j) = -m × |i - j|

    Very simple yet effective for long context with minimal parameters.

    Example:
        >>> alibi = ALiBiPositionalBias(num_heads=8)
        >>> attn_scores = np.random.randn(batch, 8, seq_len, seq_len)
        >>> biased_scores = alibi(attn_scores, seq_length=seq_len)

    Attributes:
        num_heads (int): Number of attention heads
        slopes (np.ndarray): Per-head slope values
    """

    def __init__(self, num_heads: int):
        """
        Initialize ALiBi.

        Args:
            num_heads: Number of attention heads

        Raises:
            ValueError: If num_heads is not positive
        """
        raise NotImplementedError(
            "TODO: Initialize ALiBi\n"
            "1. Call super().__init__()\n"
            "2. Validate num_heads > 0\n"
            "3. Store num_heads\n"
            "4. Compute slopes for each head:\n"
            "   - Slopes are geometric sequence: 2^(-8/n), 2^(-16/n), ...\n"
            "   - ratio = 2 ** (-8 / num_heads)\n"
            "   - slopes = ratio ** np.arange(1, num_heads + 1)\n"
            "5. Register slopes as buffer"
        )

    def forward(
        self,
        attention_scores: np.ndarray,
        seq_length: int,
        offset: int = 0,
    ) -> np.ndarray:
        """
        Apply ALiBi biases to attention scores.

        Args:
            attention_scores: Shape (batch, num_heads, seq_len, seq_len)
            seq_length: Sequence length
            offset: Position offset for KV cache

        Returns:
            Attention scores with ALiBi biases added
        """
        raise NotImplementedError(
            "TODO: Apply ALiBi biases\n"
            "1. Create position indices:\n"
            "   query_pos = np.arange(seq_length) + offset\n"
            "   key_pos = np.arange(seq_length) + offset\n"
            "2. Compute distances: dist = |query_pos[:, None] - key_pos[None, :]|\n"
            "3. Compute biases: bias = -slopes[:, None, None] * dist\n"
            "4. Return attention_scores + bias"
        )

    @staticmethod
    def get_slopes(num_heads: int) -> np.ndarray:
        """
        Compute ALiBi slopes for given number of heads.

        The slopes follow a geometric sequence that works well empirically.

        Args:
            num_heads: Number of attention heads

        Returns:
            Array of shape (num_heads,) with slope values
        """
        raise NotImplementedError(
            "TODO: Compute ALiBi slopes\n"
            "1. ratio = 2 ** (-8 / num_heads)\n"
            "2. slopes = ratio ** np.arange(1, num_heads + 1)\n"
            "3. Return slopes"
        )

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"num_heads={self.num_heads}"


# =============================================================================
# Comparison Utility
# =============================================================================

def compare_positional_encodings():
    """
    Compare different positional encoding methods.

    Returns a summary of pros/cons for each method.
    """
    comparison = {
        "Sinusoidal": {
            "pros": [
                "No learnable parameters",
                "Can extrapolate to longer sequences",
                "Deterministic"
            ],
            "cons": [
                "May be less expressive than learned",
                "Fixed pattern"
            ],
            "use_cases": ["Original Transformer", "When extrapolation needed"]
        },
        "Learned": {
            "pros": [
                "More expressive",
                "Can learn task-specific patterns"
            ],
            "cons": [
                "Cannot extrapolate beyond training length",
                "Requires more parameters"
            ],
            "use_cases": ["BERT", "GPT-2", "Fixed-length tasks"]
        },
        "Relative": {
            "pros": [
                "Captures relative positions",
                "Better for variable-length sequences"
            ],
            "cons": [
                "More complex implementation",
                "Needs bucketing for efficiency"
            ],
            "use_cases": ["Transformer-XL", "DeBERTa"]
        },
        "RoPE": {
            "pros": [
                "Excellent extrapolation",
                "Naturally encodes relative positions",
                "Works well with long context"
            ],
            "cons": [
                "More complex than sinusoidal",
                "Requires even dimensions"
            ],
            "use_cases": ["Modern LLMs (GPT-3.5, Claude, LLaMA)"]
        },
        "ALiBi": {
            "pros": [
                "Very simple",
                "Good extrapolation",
                "Minimal parameters"
            ],
            "cons": [
                "Linear bias may be limiting",
                "Less expressive"
            ],
            "use_cases": ["BLOOM", "MPT", "When simplicity matters"]
        }
    }
    return comparison
