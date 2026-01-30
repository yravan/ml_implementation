import numpy as np
import pytest
from transformers.cross_attention import (
    cross_attention,
    cross_attention_single_head,
    create_cross_attention_mask,
)


class TestCrossAttention:
    """Tests for multi-head cross-attention."""

    def test_output_shape(self):
        """Test that output shape matches query sequence length."""
        n_q, n_kv, d = 5, 10, 32
        n_heads = 4

        X_q = np.random.randn(n_q, d)
        X_kv = np.random.randn(n_kv, d)
        W_q = np.random.randn(d, d)
        W_k = np.random.randn(d, d)
        W_v = np.random.randn(d, d)
        W_o = np.random.randn(d, d)

        output, weights = cross_attention(X_q, X_kv, W_q, W_k, W_v, W_o, n_heads)

        # Output should have query sequence length
        assert output.shape == (n_q, d)
        # Attention weights should be (n_q, n_kv) for each head
        assert len(weights) == n_heads
        for w in weights:
            assert w.shape == (n_q, n_kv)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 for each query."""
        n_q, n_kv, d = 6, 8, 32
        n_heads = 4

        X_q = np.random.randn(n_q, d)
        X_kv = np.random.randn(n_kv, d)
        W_q = np.random.randn(d, d)
        W_k = np.random.randn(d, d)
        W_v = np.random.randn(d, d)
        W_o = np.random.randn(d, d)

        _, weights = cross_attention(X_q, X_kv, W_q, W_k, W_v, W_o, n_heads)

        for h, w in enumerate(weights):
            np.testing.assert_array_almost_equal(
                w.sum(axis=-1), np.ones(n_q),
                err_msg=f"Head {h} weights don't sum to 1"
            )

    def test_different_sequence_lengths(self):
        """Test cross-attention with various sequence length combinations."""
        d, n_heads = 32, 4

        test_cases = [
            (3, 10),   # Short query, long kv
            (10, 3),   # Long query, short kv
            (5, 5),    # Equal lengths
            (1, 20),   # Single query
            (20, 1),   # Single kv
        ]

        for n_q, n_kv in test_cases:
            X_q = np.random.randn(n_q, d)
            X_kv = np.random.randn(n_kv, d)
            W_q = np.random.randn(d, d) * 0.1
            W_k = np.random.randn(d, d) * 0.1
            W_v = np.random.randn(d, d) * 0.1
            W_o = np.random.randn(d, d) * 0.1

            output, weights = cross_attention(X_q, X_kv, W_q, W_k, W_v, W_o, n_heads)

            assert output.shape == (n_q, d), f"Failed for n_q={n_q}, n_kv={n_kv}"
            for w in weights:
                assert w.shape == (n_q, n_kv)

    def test_with_mask(self):
        """Test cross-attention with masking."""
        n_q, n_kv, d = 4, 6, 32
        n_heads = 2

        X_q = np.random.randn(n_q, d)
        X_kv = np.random.randn(n_kv, d)
        W_q = np.random.randn(d, d) * 0.1
        W_k = np.random.randn(d, d) * 0.1
        W_v = np.random.randn(d, d) * 0.1
        W_o = np.random.randn(d, d) * 0.1

        # Mask: only attend to first 3 kv positions
        mask = np.zeros((n_q, n_kv), dtype=bool)
        mask[:, :3] = True

        _, weights = cross_attention(X_q, X_kv, W_q, W_k, W_v, W_o, n_heads, mask=mask)

        for w in weights:
            # Masked positions should have zero weight
            np.testing.assert_array_almost_equal(w[:, 3:], 0)
            # Unmasked should sum to 1
            np.testing.assert_array_almost_equal(w.sum(axis=-1), 1)

    def test_encoder_decoder_scenario(self):
        """Test realistic encoder-decoder cross-attention scenario."""
        # Encoder: 10 source tokens
        # Decoder: generating 7 target tokens
        n_encoder, n_decoder, d = 10, 7, 64
        n_heads = 8

        encoder_output = np.random.randn(n_encoder, d)
        decoder_state = np.random.randn(n_decoder, d)

        W_q = np.random.randn(d, d) * 0.02
        W_k = np.random.randn(d, d) * 0.02
        W_v = np.random.randn(d, d) * 0.02
        W_o = np.random.randn(d, d) * 0.02

        output, weights = cross_attention(
            decoder_state, encoder_output,
            W_q, W_k, W_v, W_o, n_heads
        )

        assert output.shape == (n_decoder, d)
        assert len(weights) == n_heads
        for w in weights:
            assert w.shape == (n_decoder, n_encoder)


class TestCrossAttentionSingleHead:
    """Tests for single-head cross-attention."""

    def test_output_shape(self):
        """Test output shape."""
        n_q, n_kv, d_k, d_v = 5, 8, 16, 32
        Q = np.random.randn(n_q, d_k)
        K = np.random.randn(n_kv, d_k)
        V = np.random.randn(n_kv, d_v)

        output, weights = cross_attention_single_head(Q, K, V)

        assert output.shape == (n_q, d_v)
        assert weights.shape == (n_q, n_kv)

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1."""
        n_q, n_kv, d_k = 4, 6, 8
        Q = np.random.randn(n_q, d_k)
        K = np.random.randn(n_kv, d_k)
        V = np.random.randn(n_kv, d_k)

        _, weights = cross_attention_single_head(Q, K, V)

        np.testing.assert_array_almost_equal(weights.sum(axis=-1), 1)


class TestCrossAttentionMask:
    """Tests for cross-attention mask creation."""

    def test_shape(self):
        """Test mask shape."""
        n_q, n_kv = 5, 10
        mask = create_cross_attention_mask(n_q, n_kv)
        assert mask.shape == (n_q, n_kv)

    def test_no_padding_all_true(self):
        """Test that mask is all True when no padding specified."""
        n_q, n_kv = 5, 10
        mask = create_cross_attention_mask(n_q, n_kv)
        assert np.all(mask == True)

    def test_with_padding_mask(self):
        """Test mask with padding in key/value sequence."""
        n_q, n_kv = 4, 8

        # Padding mask: first 5 positions valid, last 3 are padding
        padding_mask_kv = np.array([True, True, True, True, True, False, False, False])

        mask = create_cross_attention_mask(n_q, n_kv, padding_mask_kv)

        assert mask.shape == (n_q, n_kv)
        # All queries should see the same padding pattern
        for i in range(n_q):
            np.testing.assert_array_equal(mask[i], padding_mask_kv)

    def test_padding_mask_blocks_attention(self):
        """Test that padding mask actually blocks attention."""
        n_q, n_kv, d_k = 3, 6, 8

        Q = np.random.randn(n_q, d_k)
        K = np.random.randn(n_kv, d_k)
        V = np.random.randn(n_kv, d_k)

        # Only first 4 kv positions are valid
        padding_mask_kv = np.array([True, True, True, True, False, False])
        mask = create_cross_attention_mask(n_q, n_kv, padding_mask_kv)

        _, weights = cross_attention_single_head(Q, K, V, mask=mask)

        # Padding positions should have zero attention
        np.testing.assert_array_almost_equal(weights[:, 4:], 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
