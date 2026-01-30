import numpy as np
import pytest
from transformers.positional_encoding import (
    sinusoidal_positional_encoding,
    learned_positional_encoding,
    rotary_positional_encoding,
    add_positional_encoding,
)


class TestSinusoidalPositionalEncoding:
    """Tests for sinusoidal positional encoding."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        n_positions, d_model = 100, 64
        pe = sinusoidal_positional_encoding(n_positions, d_model)
        assert pe.shape == (n_positions, d_model)

    def test_values_bounded(self):
        """Test that values are bounded in [-1, 1] (sin/cos range)."""
        pe = sinusoidal_positional_encoding(50, 32)
        assert np.all(pe >= -1) and np.all(pe <= 1)

    def test_unique_positions(self):
        """Test that each position has a unique encoding."""
        pe = sinusoidal_positional_encoding(100, 64)

        # Each row should be unique
        for i in range(len(pe)):
            for j in range(i + 1, len(pe)):
                assert not np.allclose(pe[i], pe[j]), f"Positions {i} and {j} are identical"

    def test_first_position_pattern(self):
        """Test the encoding at position 0."""
        pe = sinusoidal_positional_encoding(10, 8)

        # At position 0: sin(0) = 0, cos(0) = 1
        # Even dimensions should be 0 (sin), odd dimensions should be 1 (cos)
        np.testing.assert_array_almost_equal(pe[0, 0::2], 0)  # sin(0) = 0
        np.testing.assert_array_almost_equal(pe[0, 1::2], 1)  # cos(0) = 1

    def test_deterministic(self):
        """Test that encoding is deterministic (no randomness)."""
        pe1 = sinusoidal_positional_encoding(50, 32)
        pe2 = sinusoidal_positional_encoding(50, 32)
        np.testing.assert_array_equal(pe1, pe2)

    def test_different_frequencies_per_dimension(self):
        """Test that different dimensions use different frequencies."""
        pe = sinusoidal_positional_encoding(100, 16)

        # Lower dimensions should have higher frequency (more oscillations)
        # Count zero crossings as a proxy for frequency
        def count_zero_crossings(arr):
            return np.sum(np.diff(np.sign(arr)) != 0)

        # Compare first and last even dimensions
        crossings_low_dim = count_zero_crossings(pe[:, 0])
        crossings_high_dim = count_zero_crossings(pe[:, -2])

        assert crossings_low_dim > crossings_high_dim, "Lower dims should have higher frequency"

    def test_relative_position_property(self):
        """Test that dot product relates to relative position."""
        pe = sinusoidal_positional_encoding(100, 64)

        # Dot product of PE(t) and PE(t+k) should be similar for same k
        k = 5
        dots_at_distance_k = []
        for t in range(50):
            dot = np.dot(pe[t], pe[t + k])
            dots_at_distance_k.append(dot)

        # All dot products at same distance should be similar
        std = np.std(dots_at_distance_k)
        assert std < 1.0, "Dot products at same relative distance should be similar"


class TestLearnedPositionalEncoding:
    """Tests for learned positional encoding."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        n_positions, d_model = 100, 64
        pe = learned_positional_encoding(n_positions, d_model)
        assert pe.shape == (n_positions, d_model)

    def test_reproducible_with_seed(self):
        """Test that same seed produces same encoding."""
        pe1 = learned_positional_encoding(50, 32, seed=42)
        pe2 = learned_positional_encoding(50, 32, seed=42)
        np.testing.assert_array_equal(pe1, pe2)

    def test_different_with_different_seed(self):
        """Test that different seeds produce different encodings."""
        pe1 = learned_positional_encoding(50, 32, seed=42)
        pe2 = learned_positional_encoding(50, 32, seed=123)
        assert not np.allclose(pe1, pe2)

    def test_reasonable_initialization_scale(self):
        """Test that initialized values are reasonably small."""
        pe = learned_positional_encoding(100, 64, seed=42)

        # Should be small values, not huge
        assert np.abs(pe).max() < 1.0, "Initial values should be small"
        assert np.abs(pe).mean() < 0.1, "Mean absolute value should be small"


class TestRotaryPositionalEncoding:
    """Tests for rotary positional encoding (RoPE)."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d = 10, 32
        x = np.random.randn(n, d)
        positions = np.arange(n)

        x_rotated = rotary_positional_encoding(x, positions)
        assert x_rotated.shape == (n, d)

    def test_preserves_norm(self):
        """Test that rotation preserves vector norms."""
        n, d = 10, 32
        x = np.random.randn(n, d)
        positions = np.arange(n)

        x_rotated = rotary_positional_encoding(x, positions)

        # Norms should be preserved (rotation is orthogonal)
        original_norms = np.linalg.norm(x, axis=-1)
        rotated_norms = np.linalg.norm(x_rotated, axis=-1)
        np.testing.assert_array_almost_equal(original_norms, rotated_norms)

    def test_zero_position_identity(self):
        """Test that position 0 doesn't change the embedding (rotation by 0)."""
        n, d = 5, 16
        x = np.random.randn(n, d)
        positions = np.zeros(n, dtype=int)

        x_rotated = rotary_positional_encoding(x, positions)

        # At position 0, cos(0)=1, sin(0)=0, so rotation is identity
        np.testing.assert_array_almost_equal(x, x_rotated)

    def test_relative_position_in_dot_product(self):
        """Test that dot product depends on relative position."""
        d = 64
        x = np.random.randn(1, d)
        y = np.random.randn(1, d)

        # Encode at positions (0, 5) and (10, 15) - same relative distance
        x_pos0 = rotary_positional_encoding(x, np.array([0]))
        y_pos5 = rotary_positional_encoding(y, np.array([5]))
        dot1 = np.dot(x_pos0.flatten(), y_pos5.flatten())

        x_pos10 = rotary_positional_encoding(x, np.array([10]))
        y_pos15 = rotary_positional_encoding(y, np.array([15]))
        dot2 = np.dot(x_pos10.flatten(), y_pos15.flatten())

        # Dot products should be equal for same relative position
        np.testing.assert_almost_equal(dot1, dot2, decimal=5)


class TestAddPositionalEncoding:
    """Tests for adding positional encoding to embeddings."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d = 10, 32
        x = np.random.randn(n, d)
        pe = sinusoidal_positional_encoding(100, d)

        x_pos = add_positional_encoding(x, pe)
        assert x_pos.shape == (n, d)

    def test_adds_correctly(self):
        """Test that positional encoding is added correctly."""
        n, d = 5, 16
        x = np.random.randn(n, d)
        pe = sinusoidal_positional_encoding(100, d)

        x_pos = add_positional_encoding(x, pe)

        expected = x + pe[:n]
        np.testing.assert_array_almost_equal(x_pos, expected)

    def test_shorter_sequence_than_max(self):
        """Test with sequence shorter than max positional encoding length."""
        max_len, d = 100, 32
        n = 20  # Shorter than max

        x = np.random.randn(n, d)
        pe = sinusoidal_positional_encoding(max_len, d)

        x_pos = add_positional_encoding(x, pe)

        assert x_pos.shape == (n, d)
        expected = x + pe[:n]
        np.testing.assert_array_almost_equal(x_pos, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
