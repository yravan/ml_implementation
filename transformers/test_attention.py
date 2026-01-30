import numpy as np
import pytest
from transformers.attention import scaled_dot_product_attention, multi_head_attention, create_causal_mask, softmax


class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention."""

    def test_output_shape(self):
        """Test that output shapes are correct."""
        n_queries, n_keys, d_k, d_v = 5, 10, 8, 16
        Q = np.random.randn(n_queries, d_k)
        K = np.random.randn(n_keys, d_k)
        V = np.random.randn(n_keys, d_v)

        output, weights = scaled_dot_product_attention(Q, K, V)

        assert output.shape == (n_queries, d_v)
        assert weights.shape == (n_queries, n_keys)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 for each query."""
        n, d_k = 6, 8
        Q = np.random.randn(n, d_k)
        K = np.random.randn(n, d_k)
        V = np.random.randn(n, d_k)

        _, weights = scaled_dot_product_attention(Q, K, V)

        np.testing.assert_array_almost_equal(weights.sum(axis=-1), np.ones(n))

    def test_attention_weights_non_negative(self):
        """Test that attention weights are non-negative (from softmax)."""
        n, d_k = 4, 8
        Q = np.random.randn(n, d_k)
        K = np.random.randn(n, d_k)
        V = np.random.randn(n, d_k)

        _, weights = scaled_dot_product_attention(Q, K, V)

        assert np.all(weights >= 0)

    def test_identical_qk_high_self_attention(self):
        """When Q and K are identical, each query should attend most to itself."""
        n, d_k = 4, 32
        # Create distinct vectors so self-attention is clearly highest
        X = np.eye(n, d_k) * 10  # Large values on diagonal
        Q = K = X
        V = np.random.randn(n, d_k)

        _, weights = scaled_dot_product_attention(Q, K, V)

        # Each query should have highest attention to its corresponding key (diagonal)
        assert np.all(np.argmax(weights, axis=-1) == np.arange(n))

    def test_scaling_by_sqrt_dk(self):
        """Test that scaling prevents extreme softmax outputs for large d_k."""
        n = 4
        d_k_small, d_k_large = 4, 256

        np.random.seed(42)
        Q_small = np.random.randn(n, d_k_small)
        K_small = np.random.randn(n, d_k_small)
        V_small = np.random.randn(n, d_k_small)

        Q_large = np.random.randn(n, d_k_large)
        K_large = np.random.randn(n, d_k_large)
        V_large = np.random.randn(n, d_k_large)

        _, weights_small = scaled_dot_product_attention(Q_small, K_small, V_small)
        _, weights_large = scaled_dot_product_attention(Q_large, K_large, V_large)

        # With proper scaling, entropy of attention shouldn't be drastically different
        def entropy(w):
            return -np.sum(w * np.log(w + 1e-10), axis=-1).mean()

        # If scaling is wrong, large d_k would have near-zero entropy (one-hot)
        assert entropy(weights_large) > 0.5  # Should have reasonable entropy

    def test_mask_blocks_attention(self):
        """Test that masked positions get zero attention weight."""
        n, d_k = 4, 8
        Q = np.random.randn(n, d_k)
        K = np.random.randn(n, d_k)
        V = np.random.randn(n, d_k)

        # Mask: query 0 can only attend to keys 0,1; query 1 can attend to all, etc.
        mask = np.array([
            [True, True, False, False],
            [True, True, True, True],
            [True, False, True, False],
            [False, False, False, True],
        ])

        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Check that masked positions have zero weight
        np.testing.assert_array_almost_equal(weights[~mask], 0)

        # Check that unmasked positions still sum to 1
        np.testing.assert_array_almost_equal(weights.sum(axis=-1), np.ones(n))

    def test_output_is_weighted_sum_of_values(self):
        """Test that output is correctly computed as weighted sum of V."""
        n_q, n_k, d_k, d_v = 3, 4, 8, 6
        Q = np.random.randn(n_q, d_k)
        K = np.random.randn(n_k, d_k)
        V = np.random.randn(n_k, d_v)

        output, weights = scaled_dot_product_attention(Q, K, V)

        # Manually compute expected output
        expected_output = weights @ V

        np.testing.assert_array_almost_equal(output, expected_output)


class TestMultiHeadAttention:
    """Tests for multi-head attention."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d = 10, 64
        n_heads = 8
        d_k = d  # Total dimension across all heads

        X = np.random.randn(n, d)
        W_q = np.random.randn(d, d_k)
        W_k = np.random.randn(d, d_k)
        W_v = np.random.randn(d, d_k)
        W_o = np.random.randn(d_k, d)

        output, weights = multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads)

        assert output.shape == (n, d)
        assert len(weights) == n_heads

    def test_attention_weights_per_head(self):
        """Test that each head produces valid attention weights."""
        n, d = 6, 32
        n_heads = 4

        X = np.random.randn(n, d)
        W_q = np.random.randn(d, d)
        W_k = np.random.randn(d, d)
        W_v = np.random.randn(d, d)
        W_o = np.random.randn(d, d)

        _, weights = multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads)

        for h, w in enumerate(weights):
            assert w.shape == (n, n), f"Head {h} has wrong shape"
            np.testing.assert_array_almost_equal(
                w.sum(axis=-1), np.ones(n),
                err_msg=f"Head {h} weights don't sum to 1"
            )

    def test_different_heads_learn_different_patterns(self):
        """Test that different heads can produce different attention patterns."""
        n, d = 5, 32
        n_heads = 4

        np.random.seed(123)
        X = np.random.randn(n, d)
        # Use different random projections to encourage different patterns
        W_q = np.random.randn(d, d)
        W_k = np.random.randn(d, d)
        W_v = np.random.randn(d, d)
        W_o = np.random.randn(d, d)

        _, weights = multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads)

        # Check that not all heads have identical attention patterns
        if len(weights) > 1:
            differences = []
            for i in range(len(weights) - 1):
                diff = np.abs(weights[i] - weights[i + 1]).mean()
                differences.append(diff)
            # At least some heads should be different
            assert max(differences) > 0.01

    def test_with_causal_mask(self):
        """Test multi-head attention with causal masking."""
        n, d = 6, 32
        n_heads = 4

        X = np.random.randn(n, d)
        W_q = np.random.randn(d, d)
        W_k = np.random.randn(d, d)
        W_v = np.random.randn(d, d)
        W_o = np.random.randn(d, d)

        mask = create_causal_mask(n)
        _, weights = multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads, mask=mask)

        # Check that no head attends to future positions
        for h, w in enumerate(weights):
            upper_triangle = np.triu(w, k=1)  # Strictly upper triangle
            np.testing.assert_array_almost_equal(
                upper_triangle, 0,
                err_msg=f"Head {h} attends to future positions"
            )


class TestCausalMask:
    """Tests for causal mask creation."""

    def test_shape(self):
        """Test that mask has correct shape."""
        n = 5
        mask = create_causal_mask(n)
        assert mask.shape == (n, n)

    def test_lower_triangular(self):
        """Test that mask is lower triangular (including diagonal)."""
        n = 6
        mask = create_causal_mask(n)

        # Lower triangle (including diagonal) should be True
        expected = np.tril(np.ones((n, n), dtype=bool))
        np.testing.assert_array_equal(mask, expected)

    def test_first_position_only_self(self):
        """Test that first position can only attend to itself."""
        n = 5
        mask = create_causal_mask(n)

        assert mask[0, 0] == True
        assert np.all(mask[0, 1:] == False)

    def test_last_position_attends_all(self):
        """Test that last position can attend to all positions."""
        n = 5
        mask = create_causal_mask(n)

        assert np.all(mask[-1, :] == True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
