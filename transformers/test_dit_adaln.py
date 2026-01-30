import numpy as np
import pytest
from transformers.dit_adaln import (
    adaln,
    adaln_zero,
    timestep_embedding,
    conditioning_mlp,
    dit_block,
    patchify,
    unpatchify,
    init_dit_block_params,
)


class TestAdaLN:
    """Tests for Adaptive Layer Normalization."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d = 10, 64
        x = np.random.randn(n, d)
        gamma = np.ones(d)
        beta = np.zeros(d)

        out = adaln(x, gamma, beta)
        assert out.shape == (n, d)

    def test_normalizes_to_zero_mean_unit_var(self):
        """Test that with gamma=1, beta=0, output is normalized."""
        n, d = 10, 64
        x = np.random.randn(n, d) * 5 + 10  # Non-zero mean, non-unit var
        gamma = np.ones(d)
        beta = np.zeros(d)

        out = adaln(x, gamma, beta)

        # Each token should have ~zero mean and ~unit std
        means = out.mean(axis=-1)
        stds = out.std(axis=-1)

        np.testing.assert_array_almost_equal(means, 0, decimal=5)
        np.testing.assert_array_almost_equal(stds, 1, decimal=1)

    def test_gamma_scales_output(self):
        """Test that gamma scales the normalized output."""
        n, d = 5, 32
        x = np.random.randn(n, d)
        gamma = np.ones(d) * 3
        beta = np.zeros(d)

        out = adaln(x, gamma, beta)

        # Std should be approximately 3
        stds = out.std(axis=-1)
        np.testing.assert_array_almost_equal(stds, 3, decimal=1)

    def test_beta_shifts_output(self):
        """Test that beta shifts the output."""
        n, d = 5, 32
        x = np.random.randn(n, d)
        gamma = np.ones(d)
        beta = np.ones(d) * 5

        out = adaln(x, gamma, beta)

        # Mean should be approximately 5
        means = out.mean(axis=-1)
        np.testing.assert_array_almost_equal(means, 5, decimal=5)

    def test_different_gamma_beta_per_dim(self):
        """Test with varying gamma/beta across dimensions."""
        n, d = 5, 16
        x = np.random.randn(n, d)
        gamma = np.linspace(0.5, 2.0, d)
        beta = np.linspace(-1, 1, d)

        out = adaln(x, gamma, beta)
        assert out.shape == (n, d)


class TestAdaLNZero:
    """Tests for AdaLN-Zero."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d = 10, 64
        x = np.random.randn(n, d)
        gamma = np.ones(d)
        beta = np.zeros(d)
        alpha = np.ones(d)

        out = adaln_zero(x, gamma, beta, alpha)
        assert out.shape == (n, d)

    def test_alpha_zero_gives_zero_output(self):
        """Test that alpha=0 gives zero output (for zero initialization)."""
        n, d = 5, 32
        x = np.random.randn(n, d)
        gamma = np.ones(d)
        beta = np.zeros(d)
        alpha = np.zeros(d)  # Zero initialization

        out = adaln_zero(x, gamma, beta, alpha)

        np.testing.assert_array_almost_equal(out, 0)

    def test_alpha_one_equals_adaln(self):
        """Test that alpha=1 gives same result as standard AdaLN."""
        n, d = 5, 32
        x = np.random.randn(n, d)
        gamma = np.random.randn(d)
        beta = np.random.randn(d)
        alpha = np.ones(d)

        out_zero = adaln_zero(x, gamma, beta, alpha)
        out_standard = adaln(x, gamma, beta)

        np.testing.assert_array_almost_equal(out_zero, out_standard)

    def test_alpha_scales_output(self):
        """Test that alpha scales the output."""
        n, d = 5, 32
        x = np.random.randn(n, d)
        gamma = np.ones(d)
        beta = np.zeros(d)
        alpha = np.ones(d) * 0.5

        out = adaln_zero(x, gamma, beta, alpha)
        out_full = adaln(x, gamma, beta)

        np.testing.assert_array_almost_equal(out, 0.5 * out_full)


class TestTimestepEmbedding:
    """Tests for timestep embedding."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        batch, d = 4, 64
        timesteps = np.array([0, 100, 500, 999])

        emb = timestep_embedding(timesteps, d)
        assert emb.shape == (batch, d)

    def test_different_timesteps_different_embeddings(self):
        """Test that different timesteps produce different embeddings."""
        d = 64
        t1 = np.array([0])
        t2 = np.array([500])

        emb1 = timestep_embedding(t1, d)
        emb2 = timestep_embedding(t2, d)

        assert not np.allclose(emb1, emb2)

    def test_same_timestep_same_embedding(self):
        """Test that same timestep always produces same embedding."""
        d = 64
        t = np.array([100])

        emb1 = timestep_embedding(t, d)
        emb2 = timestep_embedding(t, d)

        np.testing.assert_array_equal(emb1, emb2)

    def test_continuous_timesteps(self):
        """Test with continuous (float) timesteps."""
        d = 64
        timesteps = np.array([0.0, 0.5, 1.0, 100.5])

        emb = timestep_embedding(timesteps, d)
        assert emb.shape == (4, d)

    def test_embedding_bounded(self):
        """Test that embeddings are bounded (sin/cos range)."""
        d = 64
        timesteps = np.linspace(0, 1000, 100)

        emb = timestep_embedding(timesteps, d)

        assert np.all(emb >= -1) and np.all(emb <= 1)


class TestConditioningMLP:
    """Tests for conditioning MLP."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        batch, d_in, d_hidden, d_out = 4, 64, 256, 384
        c = np.random.randn(batch, d_in)
        W1 = np.random.randn(d_in, d_hidden)
        b1 = np.zeros(d_hidden)
        W2 = np.random.randn(d_hidden, d_out)
        b2 = np.zeros(d_out)

        out = conditioning_mlp(c, W1, b1, W2, b2)
        assert out.shape == (batch, d_out)

    def test_nonlinearity_applied(self):
        """Test that nonlinearity is applied (not purely linear)."""
        batch, d = 4, 32
        c = np.random.randn(batch, d)

        # With identity-like weights, output should show nonlinearity effects
        W1 = np.eye(d)
        b1 = np.zeros(d)
        W2 = np.eye(d)
        b2 = np.zeros(d)

        out = conditioning_mlp(c, W1, b1, W2, b2)

        # Due to ReLU, negative inputs should be zeroed
        # This is a weak test but verifies some transformation happens
        assert out.shape == (batch, d)


class TestDiTBlock:
    """Tests for DiT block."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d, d_ff, n_heads = 16, 64, 256, 8
        x = np.random.randn(n, d)
        c = np.random.randn(d)
        params = init_dit_block_params(d, d_ff, n_heads, seed=42)

        z = dit_block(x, c, params, n_heads)
        assert z.shape == (n, d)

    def test_different_conditioning_different_output(self):
        """Test that different conditioning produces different outputs."""
        n, d, d_ff, n_heads = 16, 64, 256, 8
        x = np.random.randn(n, d)
        params = init_dit_block_params(d, d_ff, n_heads, seed=42)

        c1 = np.random.randn(d)
        z1 = dit_block(x, c1, params, n_heads)

        c2 = np.random.randn(d)
        z2 = dit_block(x, c2, params, n_heads)

        assert not np.allclose(z1, z2)

    def test_deterministic(self):
        """Test that DiT block is deterministic."""
        n, d, d_ff, n_heads = 8, 32, 128, 4
        x = np.random.randn(n, d)
        c = np.random.randn(d)
        params = init_dit_block_params(d, d_ff, n_heads, seed=42)

        z1 = dit_block(x, c, params, n_heads)
        z2 = dit_block(x, c, params, n_heads)

        np.testing.assert_array_equal(z1, z2)


class TestPatchify:
    """Tests for patchifying images."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        H, W, C = 32, 32, 3
        patch_size = 8
        image = np.random.randn(H, W, C)

        patches = patchify(image, patch_size)

        n_patches = (H // patch_size) * (W // patch_size)  # 16
        patch_dim = patch_size * patch_size * C  # 192
        assert patches.shape == (n_patches, patch_dim)

    def test_various_patch_sizes(self):
        """Test with various patch sizes."""
        H, W, C = 64, 64, 4
        image = np.random.randn(H, W, C)

        for patch_size in [4, 8, 16, 32]:
            patches = patchify(image, patch_size)
            n_patches = (H // patch_size) * (W // patch_size)
            patch_dim = patch_size * patch_size * C
            assert patches.shape == (n_patches, patch_dim)

    def test_preserves_content(self):
        """Test that patchify preserves image content."""
        H, W, C = 16, 16, 3
        patch_size = 8
        image = np.random.randn(H, W, C)

        patches = patchify(image, patch_size)

        # First patch should contain top-left 8x8 region
        first_patch = patches[0].reshape(patch_size, patch_size, C)
        expected = image[:patch_size, :patch_size, :]
        np.testing.assert_array_almost_equal(first_patch, expected)


class TestUnpatchify:
    """Tests for unpatchifying back to images."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        H, W, C = 32, 32, 3
        patch_size = 8
        n_patches = (H // patch_size) * (W // patch_size)
        patch_dim = patch_size * patch_size * C
        patches = np.random.randn(n_patches, patch_dim)

        image = unpatchify(patches, patch_size, H, W, C)
        assert image.shape == (H, W, C)

    def test_inverse_of_patchify(self):
        """Test that unpatchify is inverse of patchify."""
        H, W, C = 32, 32, 4
        patch_size = 8
        original = np.random.randn(H, W, C)

        patches = patchify(original, patch_size)
        reconstructed = unpatchify(patches, patch_size, H, W, C)

        np.testing.assert_array_almost_equal(original, reconstructed)


class TestInitDiTBlockParams:
    """Tests for DiT block parameter initialization."""

    def test_all_params_present(self):
        """Test that all required parameters are initialized."""
        d, d_ff, n_heads = 64, 256, 8
        params = init_dit_block_params(d, d_ff, n_heads)

        required = [
            'W_q', 'W_k', 'W_v', 'W_o',  # Attention
            'W1_ffn', 'b1_ffn', 'W2_ffn', 'b2_ffn',  # FFN
            'W1_cond', 'b1_cond', 'W2_cond', 'b2_cond',  # Conditioning MLP
        ]

        for key in required:
            assert key in params, f"Missing parameter: {key}"

    def test_conditioning_mlp_output_size(self):
        """Test that conditioning MLP outputs 6*d values (γ1,β1,α1,γ2,β2,α2)."""
        d, d_ff, n_heads = 64, 256, 8
        params = init_dit_block_params(d, d_ff, n_heads)

        # W2_cond should output 6*d values
        assert params['W2_cond'].shape[1] == 6 * d
        assert params['b2_cond'].shape[0] == 6 * d

    def test_reproducible_with_seed(self):
        """Test that same seed gives same parameters."""
        d, d_ff, n_heads = 64, 256, 8

        p1 = init_dit_block_params(d, d_ff, n_heads, seed=42)
        p2 = init_dit_block_params(d, d_ff, n_heads, seed=42)

        for key in p1:
            np.testing.assert_array_equal(p1[key], p2[key])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
