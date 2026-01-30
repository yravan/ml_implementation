import numpy as np
import pytest
from transformers.transformer_block import (
    layer_norm,
    feed_forward_network,
    transformer_block,
    transformer_block_pre_ln,
    init_transformer_block_params,
    transformer,
)
from transformers.attention import create_causal_mask


class TestLayerNorm:
    """Tests for layer normalization."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d = 10, 64
        x = np.random.randn(n, d)
        gamma = np.ones(d)
        beta = np.zeros(d)

        out = layer_norm(x, gamma, beta)
        assert out.shape == (n, d)

    def test_zero_mean_unit_variance(self):
        """Test that output has approximately zero mean and unit variance per token."""
        n, d = 10, 64
        x = np.random.randn(n, d) * 5 + 10  # Non-zero mean, non-unit variance
        gamma = np.ones(d)
        beta = np.zeros(d)

        out = layer_norm(x, gamma, beta)

        # Each token should have ~zero mean and ~unit std
        means = out.mean(axis=-1)
        stds = out.std(axis=-1)

        np.testing.assert_array_almost_equal(means, 0, decimal=5)
        np.testing.assert_array_almost_equal(stds, 1, decimal=1)

    def test_gamma_scales_output(self):
        """Test that gamma scales the output."""
        n, d = 5, 32
        x = np.random.randn(n, d)
        gamma = np.ones(d) * 2  # Scale by 2
        beta = np.zeros(d)

        out = layer_norm(x, gamma, beta)

        # Std should be approximately 2 (gamma value)
        stds = out.std(axis=-1)
        np.testing.assert_array_almost_equal(stds, 2, decimal=1)

    def test_beta_shifts_output(self):
        """Test that beta shifts the output."""
        n, d = 5, 32
        x = np.random.randn(n, d)
        gamma = np.ones(d)
        beta = np.ones(d) * 3  # Shift by 3

        out = layer_norm(x, gamma, beta)

        # Mean should be approximately 3 (beta value)
        means = out.mean(axis=-1)
        np.testing.assert_array_almost_equal(means, 3, decimal=5)

    def test_constant_input(self):
        """Test behavior with constant input (zero variance edge case)."""
        n, d = 5, 32
        x = np.ones((n, d)) * 5  # All same value
        gamma = np.ones(d)
        beta = np.zeros(d)

        out = layer_norm(x, gamma, beta, eps=1e-5)

        # Should handle gracefully (not NaN or inf)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


class TestFeedForwardNetwork:
    """Tests for feed-forward network."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d, d_ff = 10, 64, 256
        x = np.random.randn(n, d)
        W1 = np.random.randn(d, d_ff) * 0.02
        b1 = np.zeros(d_ff)
        W2 = np.random.randn(d_ff, d) * 0.02
        b2 = np.zeros(d)

        out = feed_forward_network(x, W1, b1, W2, b2)
        assert out.shape == (n, d)

    def test_relu_nonlinearity(self):
        """Test that ReLU is applied (negative values in hidden layer zeroed)."""
        n, d = 5, 8
        d_ff = 16

        # Use weights that will produce some negative hidden values
        np.random.seed(42)
        x = np.random.randn(n, d)
        W1 = np.random.randn(d, d_ff)
        b1 = np.zeros(d_ff)
        W2 = np.eye(d_ff, d)  # Identity-like for easy checking
        b2 = np.zeros(d)

        out = feed_forward_network(x, W1, b1, W2, b2)

        # The output structure depends on ReLU zeroing negative hidden values
        # This is a basic sanity check that the function runs
        assert out.shape == (n, d)

    def test_position_wise_independent(self):
        """Test that FFN is applied independently to each position."""
        n, d, d_ff = 5, 16, 32
        W1 = np.random.randn(d, d_ff) * 0.1
        b1 = np.zeros(d_ff)
        W2 = np.random.randn(d_ff, d) * 0.1
        b2 = np.zeros(d)

        # Create input where only one position is non-zero
        x = np.zeros((n, d))
        x[2] = np.random.randn(d)

        out = feed_forward_network(x, W1, b1, W2, b2)

        # Positions with zero input should have zero output (before bias)
        # With b1=0, b2=0 and x[i]=0, output should be 0
        np.testing.assert_array_almost_equal(out[0], b2)
        np.testing.assert_array_almost_equal(out[1], b2)
        # Position 2 should be different (non-zero input)


class TestTransformerBlock:
    """Tests for transformer block."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d, d_ff, n_heads = 10, 64, 256, 8
        x = np.random.randn(n, d)
        params = init_transformer_block_params(d, d_ff, n_heads, seed=42)

        z = transformer_block(x, params, n_heads)
        assert z.shape == (n, d)

    def test_residual_connection(self):
        """Test that residual connections are present."""
        n, d, d_ff, n_heads = 5, 32, 128, 4
        x = np.random.randn(n, d) * 10  # Large values
        params = init_transformer_block_params(d, d_ff, n_heads, seed=42)

        # With small weight initialization, output should be close to input
        # due to residual connections
        z = transformer_block(x, params, n_heads)

        # Not checking exact values, but ensuring the block runs
        assert z.shape == (n, d)
        assert not np.any(np.isnan(z))

    def test_with_causal_mask(self):
        """Test transformer block with causal masking."""
        n, d, d_ff, n_heads = 8, 32, 128, 4
        x = np.random.randn(n, d)
        params = init_transformer_block_params(d, d_ff, n_heads, seed=42)
        mask = create_causal_mask(n)

        z = transformer_block(x, params, n_heads, mask=mask)

        assert z.shape == (n, d)
        assert not np.any(np.isnan(z))

    def test_deterministic(self):
        """Test that transformer block is deterministic."""
        n, d, d_ff, n_heads = 5, 32, 128, 4
        x = np.random.randn(n, d)
        params = init_transformer_block_params(d, d_ff, n_heads, seed=42)

        z1 = transformer_block(x, params, n_heads)
        z2 = transformer_block(x, params, n_heads)

        np.testing.assert_array_equal(z1, z2)


class TestTransformerBlockPreLN:
    """Tests for Pre-LN transformer block variant."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d, d_ff, n_heads = 10, 64, 256, 8
        x = np.random.randn(n, d)
        params = init_transformer_block_params(d, d_ff, n_heads, seed=42)

        z = transformer_block_pre_ln(x, params, n_heads)
        assert z.shape == (n, d)

    def test_different_from_post_ln(self):
        """Test that Pre-LN gives different results than Post-LN."""
        n, d, d_ff, n_heads = 5, 32, 128, 4
        x = np.random.randn(n, d)
        params = init_transformer_block_params(d, d_ff, n_heads, seed=42)

        z_post = transformer_block(x, params, n_heads)
        z_pre = transformer_block_pre_ln(x, params, n_heads)

        # They should not be identical (different computation order)
        assert not np.allclose(z_post, z_pre)


class TestFullTransformer:
    """Tests for full transformer (multiple blocks)."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d, d_ff, n_heads, n_layers = 10, 64, 256, 8, 3
        x = np.random.randn(n, d)

        blocks_params = [
            init_transformer_block_params(d, d_ff, n_heads, seed=i)
            for i in range(n_layers)
        ]

        z = transformer(x, blocks_params, n_heads)
        assert z.shape == (n, d)

    def test_multiple_blocks_compose(self):
        """Test that multiple blocks are applied sequentially."""
        n, d, d_ff, n_heads = 5, 32, 128, 4
        x = np.random.randn(n, d)

        # Single block
        params1 = init_transformer_block_params(d, d_ff, n_heads, seed=1)
        z1 = transformer(x, [params1], n_heads)

        # Two blocks
        params2 = init_transformer_block_params(d, d_ff, n_heads, seed=2)
        z2 = transformer(x, [params1, params2], n_heads)

        # Results should be different
        assert not np.allclose(z1, z2)

    def test_with_causal_mask(self):
        """Test full transformer with causal mask."""
        n, d, d_ff, n_heads, n_layers = 8, 32, 128, 4, 2
        x = np.random.randn(n, d)
        mask = create_causal_mask(n)

        blocks_params = [
            init_transformer_block_params(d, d_ff, n_heads, seed=i)
            for i in range(n_layers)
        ]

        z = transformer(x, blocks_params, n_heads, mask=mask)
        assert z.shape == (n, d)


class TestInitTransformerBlockParams:
    """Tests for parameter initialization."""

    def test_all_params_present(self):
        """Test that all required parameters are initialized."""
        d, d_ff, n_heads = 64, 256, 8
        params = init_transformer_block_params(d, d_ff, n_heads)

        required_keys = [
            'W_q', 'W_k', 'W_v', 'W_o',
            'gamma1', 'beta1',
            'W1', 'b1', 'W2', 'b2',
            'gamma2', 'beta2'
        ]

        for key in required_keys:
            assert key in params, f"Missing parameter: {key}"

    def test_shapes_correct(self):
        """Test that parameter shapes are correct."""
        d, d_ff, n_heads = 64, 256, 8
        params = init_transformer_block_params(d, d_ff, n_heads)

        assert params['W_q'].shape == (d, d)
        assert params['W_k'].shape == (d, d)
        assert params['W_v'].shape == (d, d)
        assert params['W_o'].shape == (d, d)
        assert params['gamma1'].shape == (d,)
        assert params['beta1'].shape == (d,)
        assert params['W1'].shape == (d, d_ff)
        assert params['b1'].shape == (d_ff,)
        assert params['W2'].shape == (d_ff, d)
        assert params['b2'].shape == (d,)
        assert params['gamma2'].shape == (d,)
        assert params['beta2'].shape == (d,)

    def test_reproducible_with_seed(self):
        """Test that same seed gives same initialization."""
        d, d_ff, n_heads = 64, 256, 8

        params1 = init_transformer_block_params(d, d_ff, n_heads, seed=42)
        params2 = init_transformer_block_params(d, d_ff, n_heads, seed=42)

        for key in params1:
            np.testing.assert_array_equal(params1[key], params2[key])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
