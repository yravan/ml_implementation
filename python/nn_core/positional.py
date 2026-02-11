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
                          (can use longer sequences with interpolation if needed)

        Raises:
            ValueError: If d_model is not positive
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Create positional encoding matrix
        pe = np.zeros((max_seq_length, d_model), dtype=np.float32)
        position = np.arange(0, max_seq_length, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)

        # Register as buffer (not trainable)
        self.register_buffer('pe', pe)

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
        if seq_length > self.max_seq_length:
            raise ValueError(
                f"seq_length ({seq_length}) exceeds max_seq_length ({self.max_seq_length})"
            )

        return self.pe[:seq_length]

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
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Last dimension of x ({x.shape[-1]}) must match d_model ({self.d_model})"
            )

        seq_length = x.shape[1]
        pe = self.get_encoding(seq_length)

        return x + pe[np.newaxis, :, :]

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"d_model={self.d_model}, max_seq_length={self.max_seq_length}"

    @staticmethod
    def compute_pe(d_model: int, seq_length: int) -> np.ndarray:
        """
        Static method to compute positional encoding matrix.

        Useful for understanding the computation step-by-step.

        Args:
            d_model: Embedding dimension
            seq_length: Sequence length

        Returns:
            Array of shape (seq_length, d_model)

        Formula:
            PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = np.zeros((seq_length, d_model), dtype=np.float32)
        position = np.arange(0, seq_length, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)

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
    Can be applied additively (like sinusoidal) or as a replacement for word embeddings.

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
                          (default: 'normal' with std=0.02)

        Raises:
            ValueError: If seq_length or d_model are not positive
            ValueError: If padding_idx is out of range
        """
        super().__init__()

        if seq_length <= 0:
            raise ValueError(f"seq_length must be positive, got {seq_length}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if padding_idx is not None and (padding_idx < 0 or padding_idx >= seq_length):
            raise ValueError(f"padding_idx must be in [0, {seq_length}), got {padding_idx}")

        self.seq_length = seq_length
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.initialization = initialization

        # Initialize embedding weights
        if initialization == "normal":
            weight = np.random.normal(0, 0.02, (seq_length, d_model))
        elif initialization == "uniform":
            weight = np.random.uniform(-0.02, 0.02, (seq_length, d_model))
        elif initialization == "xavier":
            limit = np.sqrt(6.0 / (seq_length + d_model))
            weight = np.random.uniform(-limit, limit, (seq_length, d_model))
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

        # Convert to Parameter
        self.pe = Parameter(weight.astype(np.float32))

        # Zero out padding index if specified
        if padding_idx is not None:
            self.pe.data[padding_idx] = 0.0

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
        # Validate range
        if np.any(position_ids < 0):
            raise ValueError("position_ids must be non-negative")
        if np.any(position_ids >= self.seq_length):
            raise ValueError(f"position_ids must be < {self.seq_length}")

        # Look up embeddings
        return self.pe.data[position_ids]

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"seq_length={self.seq_length}, d_model={self.d_model}, initialization={self.initialization}"

    @staticmethod
    def from_pretrained(
        weights: np.ndarray,
        padding_idx: Optional[int] = None,
        freeze: bool = False,
    ) -> "LearnedPositionalEmbedding":
        """
        Create from pretrained weights.

        Useful for loading from checkpoint or transfer learning.

        Args:
            weights: Pretrained embedding weights of shape (seq_length, d_model)
            padding_idx: Optional padding index
            freeze: If True, don't allow gradient updates

        Returns:
            LearnedPositionalEmbedding initialized with given weights
        """
        seq_length, d_model = weights.shape

        # Create instance with dummy initialization
        instance = LearnedPositionalEmbedding(
            seq_length=seq_length,
            d_model=d_model,
            padding_idx=padding_idx,
            initialization="normal"
        )

        # Replace with pretrained weights
        instance.pe.data = weights.astype(np.float32)

        # Freeze if requested
        if freeze:
            instance.pe.requires_grad = False

        return instance


class RelativePositionalEmbedding(Module):
    """
    Relative position embeddings for self-attention.

    Instead of absolute positions, some models (e.g., Transformer-XL, DeBERTa)
    use relative position biases between query and key positions.

    This is more suitable for longer sequences as relative positions are bounded.

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
            max_distance: Maximum distance to bucket (beyond is assigned to max bucket)
            bidirectional: If True, use separate embeddings for ±relative_pos

        Raises:
            ValueError: If num_buckets or d_model not positive
        """
        super().__init__()

        if num_buckets <= 0:
            raise ValueError(f"num_buckets must be positive, got {num_buckets}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.num_buckets = num_buckets
        self.d_model = d_model
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        vocab_size = 2 * num_buckets if bidirectional else num_buckets

        # Initialize embedding weights
        std = d_model ** -0.5
        weight = np.random.normal(0, std, (vocab_size, d_model))
        self.embedding = Parameter(weight.astype(np.float32))

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
        # Create position indices
        query_pos = np.arange(seq_length)[:, None]
        key_pos = np.arange(seq_length)[None, :]

        # Compute relative distances
        relative_pos = key_pos - query_pos

        # Get bucket indices
        bucket_ids = self._get_position_buckets(
            relative_pos, self.num_buckets, self.max_distance, self.bidirectional
        )

        # Look up embeddings
        return self.embedding.data[bucket_ids]

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"num_buckets={self.num_buckets}, d_model={self.d_model}, max_distance={self.max_distance}, bidirectional={self.bidirectional}"

    @staticmethod
    def _get_position_buckets(
        relative_pos: np.ndarray,
        num_buckets: int,
        max_distance: int,
        bidirectional: bool = True,
    ) -> np.ndarray:
        """
        Map relative positions to bucket indices.

        Positions far apart are mapped to same bucket for memory efficiency.

        Args:
            relative_pos: Relative positions (can be negative)
            num_buckets: Number of buckets
            max_distance: Distances beyond this use max bucket
            bidirectional: Whether to handle negative positions separately

        Returns:
            Array of bucket indices (same shape as relative_pos)
        """
        relative_pos = np.array(relative_pos, dtype=np.int32)
        buckets = np.zeros_like(relative_pos, dtype=np.int32)

        if bidirectional:
            # Handle negative positions
            neg_mask = relative_pos < 0
            neg_dist = np.minimum(np.abs(relative_pos[neg_mask]) - 1, max_distance - 1)
            buckets[neg_mask] = num_buckets // 2 - 1 - neg_dist

            # Handle positive positions
            pos_mask = relative_pos >= 0
            pos_dist = np.minimum(relative_pos[pos_mask], max_distance - 1)
            buckets[pos_mask] = num_buckets // 2 + pos_dist
        else:
            # Unidirectional: only positive positions
            pos_dist = np.minimum(np.abs(relative_pos), max_distance - 1)
            buckets = pos_dist

        # Ensure buckets are in valid range
        buckets = np.clip(buckets, 0, num_buckets - 1)
        return buckets


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================

class RotaryPositionalEmbedding(Module):
    """
    Rotary Position Embedding (RoPE) for modern LLMs.

    RoPE applies rotation matrices to embedding space to encode position information.
    It's the standard for modern large language models due to excellent extrapolation
    properties and relative position awareness.

    Key insight: dot product of rotated vectors naturally encodes relative position:
    ⟨f(q, m), f(k, n)⟩ = ⟨f(q), f(k)⟩ + g(|m-n|)

    Example:
        >>> rope = RotaryPositionalEmbedding(d_model=128)
        >>> q = np.random.randn(batch_size, seq_len, 128)
        >>> k = np.random.randn(batch_size, seq_len, 128)
        >>> q_rotated, k_rotated = rope(q, k, seq_len=seq_len)

    Attributes:
        d_model (int): Embedding dimension (must be even)
        base (float): Frequency base (10000.0 is standard)
        max_seq_length (int): Max sequence length to pre-compute for efficiency
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
            base: Frequency base (10000.0 is standard from Transformer)
            max_seq_length: Maximum sequence length to cache rotations

        Raises:
            ValueError: If d_model is odd
            ValueError: If base <= 0
        """
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")

        self.d_model = d_model
        self.base = base
        self.max_seq_length = max_seq_length

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))
        self.register_buffer('inv_freq', inv_freq.astype(np.float32))

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
            seq_length: Sequence length (typically q.shape[-2])
            offset: Position offset for non-causal attention or batched inference

        Returns:
            Tuple of (q_rotated, k_rotated) with same shapes as inputs

        Raises:
            ValueError: If q.shape[-1] != d_model
            ValueError: If k.shape[-1] != d_model
            ValueError: If q.shape[-2] != k.shape[-2]

        Example:
            >>> rope = RotaryPositionalEmbedding(d_model=128)
            >>> batch_size, seq_len = 32, 1024
            >>> q = np.random.randn(batch_size, seq_len, 128)
            >>> k = np.random.randn(batch_size, seq_len, 128)
            >>> q_rot, k_rot = rope(q, k, seq_len)
            >>> assert q_rot.shape == q.shape
            >>> assert k_rot.shape == k.shape
        """
        if q.shape[-1] != self.d_model:
            raise ValueError(f"q last dimension {q.shape[-1]} != d_model {self.d_model}")
        if k.shape[-1] != self.d_model:
            raise ValueError(f"k last dimension {k.shape[-1]} != d_model {self.d_model}")
        if q.shape[-2] != k.shape[-2]:
            raise ValueError(f"q and k seq_length mismatch: {q.shape[-2]} != {k.shape[-2]}")

        # Create position indices
        m = np.arange(seq_length, dtype=np.float32) + offset

        # Compute angles: outer product of positions and inverse frequencies
        angles = np.outer(m, self.inv_freq)

        # Compute cos and sin
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # Apply rotations
        q_rotated = self._apply_rotations(q, cos_angles, sin_angles)
        k_rotated = self._apply_rotations(k, cos_angles, sin_angles)

        return q_rotated, k_rotated

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"d_model={self.d_model}, base={self.base}, max_seq_length={self.max_seq_length}"

    @staticmethod
    def _apply_rotations(
        x: np.ndarray,
        cos_angles: np.ndarray,
        sin_angles: np.ndarray,
    ) -> np.ndarray:
        """
        Apply rotation matrices to embeddings.

        Helper function to apply 2D rotations to each dimension pair.

        Args:
            x: Array of shape (..., seq_length, d_model)
            cos_angles: Array of shape (seq_length, d_model//2) - cosines
            sin_angles: Array of shape (seq_length, d_model//2) - sines

        Returns:
            Rotated array of shape (..., seq_length, d_model)

        Note:
            Operates on dimension pairs:
            (x[..., 2i], x[..., 2i+1]) -> rotated pair
        """
        # Get original shape
        *batch_dims, seq_length, d_model = x.shape

        # Reshape to group pairs
        x_reshaped = x.reshape(*batch_dims, seq_length, d_model // 2, 2)

        # Extract even and odd dimensions
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]

        # Apply rotation
        x_rot_even = x_even * cos_angles - x_odd * sin_angles
        x_rot_odd = x_even * sin_angles + x_odd * cos_angles

        # Stack back together
        x_rotated = np.stack([x_rot_even, x_rot_odd], axis=-1)

        # Reshape back to original shape
        x_rotated = x_rotated.reshape(*batch_dims, seq_length, d_model)

        return x_rotated

    @staticmethod
    def compute_inverse_frequencies(
        d_model: int, base: float = 10000.0
    ) -> np.ndarray:
        """
        Compute inverse frequencies for RoPE.

        Static method for understanding the frequency computation.

        Args:
            d_model: Embedding dimension (must be even)
            base: Frequency base (10000.0 standard)

        Returns:
            Array of shape (d_model // 2,) with inverse frequencies

        Formula:
            inv_freq[i] = 1 / (base^(2i / d_model)) for i = 0, 1, ..., d_model/2 - 1
        """
        dim_indices = np.arange(0, d_model, 2, dtype=np.float32)
        exponents = 2 * dim_indices / d_model
        frequencies = base ** exponents
        inv_freq = 1.0 / frequencies

        return inv_freq

    def interpolate_for_longer_sequences(
        self, new_max_seq_length: int, scaling_factor: Optional[float] = None
    ) -> None:
        """
        Extend RoPE to support longer sequences via interpolation.

        For models trained on seq_len=2048 that need seq_len=8192,
        can use linear interpolation by scaling positions.

        Args:
            new_max_seq_length: New maximum sequence length to support
            scaling_factor: Linear scaling factor. If None, computed automatically

        Note:
            Simple linear interpolation works surprisingly well!
            position_scaled = position * (training_seq_len / inference_seq_len)
        """
        if scaling_factor is None:
            scaling_factor = new_max_seq_length / self.max_seq_length

        self.max_seq_length = new_max_seq_length
        self.scaling_factor = scaling_factor


def create_rope_encoding(
    seq_length: int,
    d_model: int,
    base: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to create RoPE cos/sin tables.

    Args:
        seq_length: Sequence length
        d_model: Embedding dimension (must be even)
        base: Frequency base

    Returns:
        Tuple of (cos_angles, sin_angles) arrays
        - cos_angles: shape (seq_length, d_model // 2)
        - sin_angles: shape (seq_length, d_model // 2)

    Example:
        >>> cos_t, sin_t = create_rope_encoding(seq_length=512, d_model=128)
        >>> assert cos_t.shape == (512, 64)
    """
    inv_freq = RotaryPositionalEmbedding.compute_inverse_frequencies(d_model, base)
    positions = np.arange(seq_length, dtype=np.float32)
    angles = np.outer(positions, inv_freq)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    return cos_angles, sin_angles


# =============================================================================
# ALiBi (Attention with Linear Biases)
# =============================================================================

class ALiBiPositionalBias(Module):
    """
    Attention with Linear Biases (ALiBi) positional encoding.

    Adds linear position-dependent biases to attention logits to encode position.
    Extremely effective for long context and length extrapolation.

    Formula:
        bias(i, j) = -m × |i - j|

    Where m is a scalar bias slope (one per attention head).

    The paper proposes:
        m_h = 1 / (8^(2h / num_heads))

    Example:
        >>> alibi = ALiBiPositionalBias(num_heads=8)
        >>> q = np.random.randn(batch_size, 8, seq_len, 64)  # 8 heads
        >>> k = np.random.randn(batch_size, 8, seq_len, 64)
        >>> scores = q @ k.transpose(0, 1, 3, 2)  # (batch, 8, seq, seq)
        >>> scores = alibi(scores, seq_len)  # Apply ALiBi biases

    Attributes:
        num_heads (int): Number of attention heads
        slopes (np.ndarray): Per-head bias slopes, shape (num_heads,)
    """

    def __init__(
        self,
        num_heads: int,
        learnable_slopes: bool = False,
        max_seq_length: int = 8192,
    ):
        """
        Initialize ALiBi positional bias.

        Args:
            num_heads: Number of attention heads
            learnable_slopes: If True, make bias slopes learnable parameters
                            Default False: slopes are fixed
            max_seq_length: Maximum sequence length (for caching bias matrices)

        Raises:
            ValueError: If num_heads <= 0
        """
        super().__init__()

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.learnable_slopes = learnable_slopes

        # Compute bias slopes
        slopes = self.compute_bias_slopes(num_heads)

        if learnable_slopes:
            self.slopes = Parameter(slopes)
        else:
            self.register_buffer('slopes', slopes)

    def forward(
        self,
        attention_scores: np.ndarray,
        seq_length: int,
        offset: int = 0,
    ) -> np.ndarray:
        """
        Apply ALiBi biases to attention scores.

        Args:
            attention_scores: Array of shape (batch_size, num_heads, seq_length, seq_length)
                            or (batch_size, seq_length, seq_length) if single-head
            seq_length: Sequence length (usually attention_scores.shape[-1])
            offset: Position offset (useful for KV cache in generation)

        Returns:
            Array with same shape as input, with ALiBi biases applied

        Raises:
            ValueError: If attention_scores doesn't match expected shape
            ValueError: If seq_length differs from actual input size

        Note:
            ALiBi biases are ADDED to attention scores before softmax:
            scores[i, j] -= bias_slope × |i - j|

            The negative sign penalizes tokens farther apart.
        """
        # Get slopes
        if isinstance(self.slopes, Parameter):
            slopes = self.slopes.data
        else:
            slopes = self.slopes

        # Compute bias matrix
        bias_matrix = self.compute_bias_matrix(seq_length, slopes, offset)

        # Handle different input shapes
        if attention_scores.ndim == 4:
            # Shape: (batch_size, num_heads, seq_length, seq_length)
            bias_matrix = bias_matrix[np.newaxis, :, :, :]  # Add batch dimension
        elif attention_scores.ndim == 3:
            # Shape: (batch_size, seq_length, seq_length) - single head
            bias_matrix = bias_matrix[0, np.newaxis, :, :]  # Use first head's bias

        return attention_scores + bias_matrix

    @staticmethod
    def compute_bias_slopes(num_heads: int) -> np.ndarray:
        """
        Compute ALiBi bias slopes for each attention head.

        Static method for understanding slope computation.

        Args:
            num_heads: Number of attention heads

        Returns:
            Array of shape (num_heads,) with bias slopes

        Formula (from paper):
            m_h = 1 / (8^(2h / num_heads)) for h = 0, 1, ..., num_heads-1

        Note:
            This creates a geometric progression of slopes:
            - Head 0 has largest slope (most position-sensitive)
            - Head num_heads-1 has smallest slope (least position-sensitive)
        """
        head_indices = np.arange(num_heads, dtype=np.float32)
        exponents = 2 * head_indices / num_heads
        bases = 8 ** exponents
        slopes = 1.0 / bases

        return slopes.astype(np.float32)

    @staticmethod
    def compute_bias_matrix(
        seq_length: int,
        slopes: np.ndarray,
        offset: int = 0,
    ) -> np.ndarray:
        """
        Compute full ALiBi bias matrix.

        Args:
            seq_length: Sequence length
            slopes: Array of shape (num_heads,) with per-head bias slopes
            offset: Position offset for batched inference with KV cache

        Returns:
            Array of shape (num_heads, seq_length, seq_length)
            result[h, i, j] = -slopes[h] × |i - j|

        Note:
            Can be precomputed and cached for efficiency.
            Only needs to be recomputed if seq_length changes.
        """
        # Create position indices
        query_pos = np.arange(seq_length, dtype=np.float32) + offset
        key_pos = np.arange(seq_length, dtype=np.float32)

        # Compute distance matrix: |query_pos[:, None] - key_pos[None, :]|
        distances = np.abs(query_pos[:, np.newaxis] - key_pos[np.newaxis, :])

        # Apply slopes: -slopes[h] * distances for each head
        num_heads = slopes.shape[0]
        bias_matrix = -slopes[:, np.newaxis, np.newaxis] * distances[np.newaxis, :, :]

        return bias_matrix.astype(np.float32)

    def get_bias_for_head(
        self, head_idx: int, seq_length: int
    ) -> np.ndarray:
        """
        Get ALiBi bias matrix for a specific attention head.

        Args:
            head_idx: Head index
            seq_length: Sequence length

        Returns:
            Bias matrix for that head, shape (seq_length, seq_length)
        """
        if isinstance(self.slopes, Parameter):
            slopes = self.slopes.data
        else:
            slopes = self.slopes

        bias_matrix = self.compute_bias_matrix(seq_length, slopes)
        return bias_matrix[head_idx]

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"num_heads={self.num_heads}, learnable_slopes={self.learnable_slopes}, max_seq_length={self.max_seq_length}"


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_positional_encodings():
    """
    Comparison utility for different positional encoding methods.

    Returns a summary of characteristics:
    - Memory usage
    - Extrapolation capability
    - Position awareness
    - Implementation complexity
    """
    comparison = {
        "SinusoidalPositionalEncoding": {
            "memory": "O(1) - no parameters",
            "extrapolation": "Good - deterministic formula",
            "position_awareness": "Moderate - absolute positions",
            "complexity": "Low",
            "best_for": "Transformer baseline, academic models",
        },
        "LearnedPositionalEmbedding": {
            "memory": "O(seq_length × d_model) - large",
            "extrapolation": "Poor - fixed max length",
            "position_awareness": "Good - task-specific",
            "complexity": "Very Low",
            "best_for": "BERT, fixed-length sequences",
        },
        "RotaryPositionalEmbedding (RoPE)": {
            "memory": "O(d_model) - very small",
            "extrapolation": "Excellent - via interpolation",
            "position_awareness": "Excellent - relative positions",
            "complexity": "Medium - rotation matrices",
            "best_for": "Modern LLMs (GPT-3.5, Claude, PaLM)",
        },
        "ALiBiPositionalBias": {
            "memory": "O(num_heads) - minimal",
            "extrapolation": "Excellent - linear bias generalizes",
            "position_awareness": "Good - head-specific slopes",
            "complexity": "Low - simple subtraction",
            "best_for": "Long context, length extrapolation",
        },
    }
    return comparison
