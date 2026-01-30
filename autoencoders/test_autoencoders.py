"""
Tests for all autoencoder variants.
"""
import numpy as np
import pytest


class TestBasicAutoencoder:
    """Tests for basic autoencoder."""

    def test_encoder_output_shape(self):
        """Encoder should map (batch, d) → (batch, k)."""
        from autoencoders.autoencoder import Encoder
        batch, d, hidden, k = 16, 784, 256, 32
        enc = Encoder(d, hidden, k)
        x = np.random.randn(batch, d)
        z = enc.forward(x)
        assert z.shape == (batch, k)

    def test_decoder_output_shape(self):
        """Decoder should map (batch, k) → (batch, d)."""
        from autoencoders.autoencoder import Decoder
        batch, d, hidden, k = 16, 784, 256, 32
        dec = Decoder(k, hidden, d)
        z = np.random.randn(batch, k)
        x_recon = dec.forward(z)
        assert x_recon.shape == (batch, d)

    def test_autoencoder_reconstruction_shape(self):
        """Full autoencoder should preserve input shape."""
        from autoencoders.autoencoder import Autoencoder
        batch, d, hidden, k = 16, 784, 256, 32
        ae = Autoencoder(d, hidden, k)
        x = np.random.randn(batch, d)
        x_recon, z = ae.forward(x)
        assert x_recon.shape == x.shape
        assert z.shape == (batch, k)

    def test_mse_loss_gradient_shape(self):
        """MSE loss gradient should match reconstruction shape."""
        from autoencoders.autoencoder import mse_loss
        batch, d = 16, 784
        x = np.random.randn(batch, d)
        x_recon = np.random.randn(batch, d)
        loss, grad = mse_loss(x_recon, x)
        assert isinstance(loss, (float, np.floating))
        assert grad.shape == x_recon.shape

    def test_mse_loss_zero_for_identical(self):
        """MSE loss should be zero when reconstruction equals input."""
        from autoencoders.autoencoder import mse_loss
        x = np.random.randn(16, 784)
        loss, _ = mse_loss(x, x)
        np.testing.assert_almost_equal(loss, 0)

    def test_bce_loss_bounded(self):
        """BCE loss should be non-negative."""
        from autoencoders.autoencoder import bce_loss
        batch, d = 16, 784
        x = np.random.rand(batch, d)  # [0, 1]
        x_recon = np.random.rand(batch, d)
        loss, grad = bce_loss(x_recon, x)
        assert loss >= 0
        assert grad.shape == x_recon.shape


class TestVAE:
    """Tests for Variational Autoencoder."""

    def test_encoder_outputs_mu_and_logvar(self):
        """VAE encoder should output μ and log(σ²)."""
        from autoencoders.vae import VAEEncoder
        batch, d, hidden, k = 16, 784, 256, 20
        enc = VAEEncoder(d, hidden, k)
        x = np.random.randn(batch, d)
        mu, logvar = enc.forward(x)
        assert mu.shape == (batch, k)
        assert logvar.shape == (batch, k)

    def test_reparameterize_shape(self):
        """Reparameterization should output same shape as mu/logvar."""
        from autoencoders.vae import reparameterize
        batch, k = 16, 20
        mu = np.random.randn(batch, k)
        logvar = np.random.randn(batch, k)
        z, eps = reparameterize(mu, logvar)
        assert z.shape == (batch, k)
        assert eps.shape == (batch, k)

    def test_reparameterize_deterministic_with_eps(self):
        """Same epsilon should give same z."""
        from autoencoders.vae import reparameterize
        batch, k = 16, 20
        mu = np.random.randn(batch, k)
        logvar = np.random.randn(batch, k)
        eps = np.random.randn(batch, k)
        z1, _ = reparameterize(mu, logvar, eps)
        z2, _ = reparameterize(mu, logvar, eps)
        np.testing.assert_array_equal(z1, z2)

    def test_kl_divergence_non_negative(self):
        """KL divergence should be non-negative."""
        from autoencoders.vae import kl_divergence_gaussian
        batch, k = 16, 20
        mu = np.random.randn(batch, k)
        logvar = np.random.randn(batch, k)
        kl, dmu, dlogvar = kl_divergence_gaussian(mu, logvar)
        assert kl >= 0

    def test_kl_divergence_zero_at_prior(self):
        """KL should be ~0 when q(z|x) = p(z) = N(0, I)."""
        from autoencoders.vae import kl_divergence_gaussian
        batch, k = 16, 20
        mu = np.zeros((batch, k))
        logvar = np.zeros((batch, k))  # log(1) = 0
        kl, _, _ = kl_divergence_gaussian(mu, logvar)
        np.testing.assert_almost_equal(kl, 0, decimal=5)

    def test_vae_forward_shapes(self):
        """VAE forward should return correct shapes."""
        from autoencoders.vae import VAE
        batch, d, hidden, k = 16, 784, 256, 20
        vae = VAE(d, hidden, k)
        x = np.random.randn(batch, d)
        x_recon, mu, logvar, z = vae.forward(x)
        assert x_recon.shape == (batch, d)
        assert mu.shape == (batch, k)
        assert logvar.shape == (batch, k)
        assert z.shape == (batch, k)

    def test_vae_sample_shape(self):
        """VAE sample should generate correct shape."""
        from autoencoders.vae import VAE
        d, hidden, k = 784, 256, 20
        vae = VAE(d, hidden, k)
        samples = vae.sample(10)
        assert samples.shape == (10, d)

    def test_vae_interpolate_shape(self):
        """VAE interpolate should return n_steps samples."""
        from autoencoders.vae import VAE
        d, hidden, k = 784, 256, 20
        vae = VAE(d, hidden, k)
        x1 = np.random.randn(1, d)
        x2 = np.random.randn(1, d)
        interp = vae.interpolate(x1, x2, n_steps=10)
        assert interp.shape == (10, d)


class TestBetaVAE:
    """Tests for β-VAE."""

    def test_loss_with_beta(self):
        """β-VAE loss should weight KL by β."""
        from autoencoders.beta_vae import BetaVAE
        d, hidden, k = 784, 256, 10
        vae_b1 = BetaVAE(d, hidden, k, beta=1.0)
        vae_b4 = BetaVAE(d, hidden, k, beta=4.0)

        x = np.random.randn(16, d)
        x_recon = np.random.randn(16, d)
        mu = np.random.randn(16, k)
        logvar = np.random.randn(16, k)

        _, _, kl1, weighted_kl1 = vae_b1.compute_loss(x, x_recon, mu, logvar)
        _, _, kl4, weighted_kl4 = vae_b4.compute_loss(x, x_recon, mu, logvar)

        # Same raw KL, but weighted KL should differ by factor of 4
        np.testing.assert_almost_equal(kl1, kl4)
        np.testing.assert_almost_equal(weighted_kl4, 4 * weighted_kl1)

    def test_traverse_latent_shape(self):
        """Latent traversal should return n_steps samples."""
        from autoencoders.beta_vae import BetaVAE
        d, hidden, k = 784, 256, 10
        vae = BetaVAE(d, hidden, k, beta=4.0)
        x = np.random.randn(1, d)
        traversals = vae.traverse_latent(x, dim=0, n_steps=10)
        assert traversals.shape == (10, d)

    def test_annealed_beta_warmup(self):
        """Annealed β should increase during warmup."""
        from autoencoders.beta_vae import AnnealedBetaVAE
        d, hidden, k = 784, 256, 10
        vae = AnnealedBetaVAE(d, hidden, k, beta_max=4.0, warmup_steps=100)

        vae.step = 0
        beta_start = vae.get_beta()

        vae.step = 50
        beta_mid = vae.get_beta()

        vae.step = 100
        beta_end = vae.get_beta()

        assert beta_start < beta_mid < beta_end
        np.testing.assert_almost_equal(beta_end, 4.0)


class TestCVAE:
    """Tests for Conditional VAE."""

    def test_one_hot_encode(self):
        """One-hot encoding should work correctly."""
        from autoencoders.cvae import one_hot_encode
        labels = np.array([0, 1, 2, 9])
        one_hot = one_hot_encode(labels, 10)
        assert one_hot.shape == (4, 10)
        assert np.all(one_hot.sum(axis=1) == 1)
        assert one_hot[0, 0] == 1
        assert one_hot[3, 9] == 1

    def test_encoder_with_condition(self):
        """CVAE encoder should accept condition."""
        from autoencoders.cvae import CVAEEncoder
        batch, d, c_dim, hidden, k = 16, 784, 10, 256, 20
        enc = CVAEEncoder(d, c_dim, hidden, k)
        x = np.random.randn(batch, d)
        c = np.random.randn(batch, c_dim)
        mu, logvar = enc.forward(x, c)
        assert mu.shape == (batch, k)
        assert logvar.shape == (batch, k)

    def test_decoder_with_condition(self):
        """CVAE decoder should accept condition."""
        from autoencoders.cvae import CVAEDecoder
        batch, d, c_dim, hidden, k = 16, 784, 10, 256, 20
        dec = CVAEDecoder(k, c_dim, hidden, d)
        z = np.random.randn(batch, k)
        c = np.random.randn(batch, c_dim)
        x_recon = dec.forward(z, c)
        assert x_recon.shape == (batch, d)

    def test_cvae_forward_shapes(self):
        """CVAE forward should return correct shapes."""
        from autoencoders.cvae import CVAE
        batch, d, c_dim, hidden, k = 16, 784, 10, 256, 20
        cvae = CVAE(d, c_dim, hidden, k)
        x = np.random.randn(batch, d)
        c = np.random.randn(batch, c_dim)
        x_recon, mu, logvar, z = cvae.forward(x, c)
        assert x_recon.shape == (batch, d)
        assert mu.shape == (batch, k)

    def test_cvae_sample_class(self):
        """CVAE should generate samples for specific class."""
        from autoencoders.cvae import CVAE
        d, c_dim, hidden, k = 784, 10, 256, 20
        cvae = CVAE(d, c_dim, hidden, k)
        samples = cvae.sample_class(5, 10, n_samples=8)
        assert samples.shape == (8, d)

    def test_cvae_interpolate_class(self):
        """CVAE should interpolate between conditions."""
        from autoencoders.cvae import CVAE, one_hot_encode
        d, c_dim, hidden, k = 784, 10, 256, 20
        cvae = CVAE(d, c_dim, hidden, k)
        x = np.random.randn(1, d)
        c1 = one_hot_encode(np.array([0]), 10)
        c2 = one_hot_encode(np.array([9]), 10)
        interp = cvae.interpolate_class(x, c1, c2, n_steps=10)
        assert interp.shape == (10, d)


class TestVQVAE:
    """Tests for VQ-VAE."""

    def test_find_nearest_codebook(self):
        """Should find nearest codebook vector."""
        from autoencoders.vqvae import find_nearest_codebook
        batch, k, n_emb = 16, 64, 512
        z_e = np.random.randn(batch, k)
        codebook = np.random.randn(n_emb, k)
        indices, z_q = find_nearest_codebook(z_e, codebook)
        assert indices.shape == (batch,)
        assert z_q.shape == (batch, k)
        assert np.all(indices >= 0) and np.all(indices < n_emb)

    def test_quantized_comes_from_codebook(self):
        """Quantized vectors should be codebook vectors."""
        from autoencoders.vqvae import find_nearest_codebook
        batch, k, n_emb = 16, 64, 512
        z_e = np.random.randn(batch, k)
        codebook = np.random.randn(n_emb, k)
        indices, z_q = find_nearest_codebook(z_e, codebook)
        # Each z_q should equal codebook[indices]
        expected = codebook[indices]
        np.testing.assert_array_almost_equal(z_q, expected)

    def test_vq_loss_components(self):
        """VQ loss should have codebook and commitment components."""
        from autoencoders.vqvae import vq_loss
        batch, k = 16, 64
        z_e = np.random.randn(batch, k)
        z_q = np.random.randn(batch, k)
        loss, cb_loss, commit_loss = vq_loss(z_e, z_q, beta=0.25)
        assert loss >= 0
        assert cb_loss >= 0
        assert commit_loss >= 0

    def test_vqvae_forward_shapes(self):
        """VQ-VAE forward should return correct shapes."""
        from autoencoders.vqvae import VQVAE
        batch, d, hidden, k, n_emb = 16, 784, 256, 64, 512
        vqvae = VQVAE(d, hidden, k, n_emb)
        x = np.random.randn(batch, d)
        x_recon, z_e, z_q, indices, vq_loss = vqvae.forward(x)
        assert x_recon.shape == (batch, d)
        assert z_e.shape == (batch, k)
        assert z_q.shape == (batch, k)
        assert indices.shape == (batch,)

    def test_vqvae_indices_valid(self):
        """VQ-VAE indices should be valid codebook indices."""
        from autoencoders.vqvae import VQVAE
        batch, d, hidden, k, n_emb = 32, 784, 256, 64, 512
        vqvae = VQVAE(d, hidden, k, n_emb)
        x = np.random.randn(batch, d)
        _, _, _, indices, _ = vqvae.forward(x)
        assert np.all(indices >= 0)
        assert np.all(indices < n_emb)

    def test_vqvae_encode_returns_discrete(self):
        """VQ-VAE encode should return discrete indices."""
        from autoencoders.vqvae import VQVAE
        batch, d, hidden, k, n_emb = 16, 784, 256, 64, 512
        vqvae = VQVAE(d, hidden, k, n_emb)
        x = np.random.randn(batch, d)
        indices = vqvae.encode(x)
        assert indices.dtype in [np.int32, np.int64, int]
        assert indices.shape == (batch,)

    def test_vqvae_decode_from_indices(self):
        """Should decode from discrete indices."""
        from autoencoders.vqvae import VQVAE
        batch, d, hidden, k, n_emb = 16, 784, 256, 64, 512
        vqvae = VQVAE(d, hidden, k, n_emb)
        indices = np.random.randint(0, n_emb, batch)
        x_recon = vqvae.decode_from_indices(indices)
        assert x_recon.shape == (batch, d)

    def test_codebook_usage_perplexity(self):
        """Codebook usage should compute perplexity."""
        from autoencoders.vqvae import VQVAE
        d, hidden, k, n_emb = 784, 256, 64, 512
        vqvae = VQVAE(d, hidden, k, n_emb)
        data = np.random.randn(100, d)
        usage, perplexity = vqvae.get_codebook_usage(data)
        assert usage.shape == (n_emb,)
        assert perplexity > 0
        assert perplexity <= n_emb  # Can't exceed codebook size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
