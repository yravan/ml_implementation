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

from . import normal_, uniform_, xavier_normal_
from .module import Module, Parameter
from ..foundations import Tensor


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
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.pe = self.compute_pe(d_model=d_model, seq_length=max_seq_length)

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
        return self.pe[:seq_length]

    def forward(self, x: Tensor) -> Tensor:
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
        seq_length = x.shape[1]
        return x + self.get_encoding(seq_length)

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
        pe = np.zeros([seq_length, d_model])
        half = d_model // 2
        freq = 1 / (10000 ** (np.arange(half) * 2 / d_model))
        table = np.arange(seq_length)[:, None] * freq[None, :] # seq_len, half
        sin_table = np.sin(table)
        cos_table = np.cos(table)
        pe[:, ::2] = sin_table
        pe[:, 1::2] = cos_table
        return pe


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
        self.d_model = d_model
        self.seq_length = seq_length
        self.padding_idx = padding_idx
        self.pe = Parameter(np.zeros([seq_length, d_model]))
        if initialization == "normal":
            normal_(self.pe)
        elif initialization == "uniform":
            uniform_(self.pe)
        elif initialization == "xavier":
            xavier_normal_(self.pe)
        else:
            raise ValueError(f"Initialization scheme {initialization} not supported")

    def forward(self, position_ids: np.ndarray) -> Tensor:
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
        return self.pe[position_ids]

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
        super().__init__()
        self.num_buckets = num_buckets
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.max_distance = max_distance
        if self.bidirectional:
            vocab_size = 2 * self.num_buckets
        else:
            vocab_size = self.num_buckets
        self.embedding = Parameter(np.zeros([vocab_size, d_model]))
        normal_(self.embedding, std=np.sqrt(d_model))

    def forward(self, seq_length: int) -> Tensor:
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
        query_pos = np.arange(seq_length)[:, None]
        key_pos = np.arange(seq_length)[None, :]
        distances = query_pos - key_pos
        buckets = self._get_position_buckets(distances, self.num_buckets, self.max_distance, self.bidirectional)
        return self.embedding[buckets]


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
        ret = 0
        n = -relative_pos
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).astype(np.int32) * num_buckets
            n = np.abs(n)
        else:
            n = np.maximum(n, 0)

        # half buckets are exact (small distances)
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # other half are log-spaced (clamp to avoid log(0))
        n_clamped = np.maximum(n, 1)
        val_if_large = max_exact + (
            np.log(n_clamped.astype(np.float32) / max_exact)
            / np.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)

        ret += np.where(is_small, n, val_if_large)
        return ret


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
        super().__init__()
        self.d_model = d_model
        self.base = base
        inv_freq = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))
        t = np.arange(max_seq_length)
        freqs = np.outer(t, inv_freq)
        self.cos = np.cos(freqs)
        self.sin = np.sin(freqs)

    def _rotate(self, x, offset: int = 0):
        """
        Apply rotary embedding to a single tensor.

        Args:
            x: Tensor of shape (..., seq_length, d_model)
            offset: Position offset for KV cache

        Returns:
            Rotated tensor with same shape
        """
        if isinstance(x, Tensor):
            data = x.data
        else:
            data = x
        L = data.shape[-2]
        pairs = data.reshape(*(data.shape[:-1] + (self.d_model // 2, 2)))
        cos = self.cos[offset:L + offset]
        sin = self.sin[offset:L + offset]
        rotated = np.empty_like(pairs)
        rotated[..., 0] = cos * pairs[..., 0] - sin * pairs[..., 1]
        rotated[..., 1] = sin * pairs[..., 0] + cos * pairs[..., 1]
        result = rotated.reshape(*data.shape)
        if isinstance(x, Tensor):
            return Tensor(result, requires_grad=x.requires_grad)
        return result

    def forward(
        self,
        q: Tensor,
        k: Tensor = None,
        seq_length: int = None,
        offset: int = 0,
    ) -> Tensor:
        """
        Apply rotary embeddings to query (and optionally key).

        Args:
            q: Query tensor of shape (..., seq_length, d_model)
            k: Optional key tensor of shape (..., seq_length, d_model)
            seq_length: Sequence length (inferred from q if not given)
            offset: Position offset for KV cache

        Returns:
            If k is provided: Tuple of (rotated_q, rotated_k)
            If k is None: rotated_q only
        """
        if seq_length is None:
            seq_length = q.shape[-2] if isinstance(q, Tensor) else q.shape[-2]
        q_rotated = self._rotate(q, offset)
        if k is not None:
            k_rotated = self._rotate(k, offset)
            return q_rotated, k_rotated
        return q_rotated

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
    half = d_model // 2
    inv_freq = 1 / (base ** (np.arange(half) / half))
    timesteps = np.arange(seq_length)
    cos_table = np.outer(timesteps, inv_freq)
    sin_table = np.outer(timesteps, -inv_freq)
    cos_table = np.cos(cos_table)
    sin_table = np.sin(sin_table)
    return cos_table, sin_table


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
        self.num_heads = num_heads
        self.slopes = self.get_slopes(num_heads)

    @staticmethod
    def get_slopes(num_heads: int) -> np.ndarray:
        """
        Compute ALiBi slopes for each attention head.

        Slopes follow a geometric sequence: 2^(-8/n), 2^(-16/n), ...

        Args:
            num_heads: Number of attention heads

        Returns:
            Array of shape (num_heads,) with slope values (decreasing)
        """
        ratio = 2 ** (-8 / num_heads)
        return ratio ** np.arange(1, num_heads + 1)

    def forward(
        self,
        attention_scores: Tensor,
        seq_length: int,
        offset: int = 0,
    ) -> Tensor:
        """
        Apply ALiBi biases to attention scores.

        ALiBi penalizes distant positions: bias(i,j) = -m * |i - j|

        Args:
            attention_scores: Shape (batch, num_heads, seq_len, seq_len)
            seq_length: Sequence length
            offset: Position offset for KV cache

        Returns:
            Attention scores with ALiBi biases added
        """
        query_pos = np.arange(seq_length)[:, None] + offset
        key_pos = np.arange(seq_length)[None, :]
        distances = np.abs(query_pos - key_pos)
        # Negative bias: penalize distant positions
        biases = -distances[None, :, :] * self.slopes[:, None, None]
        return attention_scores + biases[None, ...]

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
