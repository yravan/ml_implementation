"""
Tests for all PyTorch autoencoder variants.

These tests verify correct shapes, loss properties, and gradient flow.
All tests should pass once the stub implementations are filled in.
"""
import torch
import pytest


class TestBasicAutoencoder:
    """Tests for basic autoencoder."""

    def test_encoder_output_shape(self):
        """Encoder should map (batch, d) -> (batch, k)."""
        from autoencoders_pytorch.autoencoder import Encoder
        batch, d, hidden, k = 16, 784, 256, 32
        enc = Encoder(d, hidden, k)
        x = torch.randn(batch, d)
        z = enc(x)
        assert z.shape == (batch, k)

    def test_decoder_output_shape(self):
        """Decoder should map (batch, k) -> (batch, d)."""
        from autoencoders_pytorch.autoencoder import Decoder
        batch, d, hidden, k = 16, 784, 256, 32
        dec = Decoder(k, hidden, d)
        z = torch.randn(batch, k)
        x_recon = dec(z)
        assert x_recon.shape == (batch, d)

    def test_autoencoder_reconstruction_shape(self):
        """Full autoencoder should preserve input shape."""
        from autoencoders_pytorch.autoencoder import Autoencoder
        batch, d, hidden, k = 16, 784, 256, 32
        ae = Autoencoder(d, hidden, k)
        x = torch.randn(batch, d)
        x_recon, z = ae(x)
        assert x_recon.shape == x.shape
        assert z.shape == (batch, k)

    def test_mse_loss(self):
        """MSE loss should be non-negative and zero for identical inputs."""
        from autoencoders_pytorch.autoencoder import Autoencoder
        ae = Autoencoder(784, 256, 32)
        x = torch.randn(16, 784)
        x_recon, _ = ae(x)
        loss = ae.compute_loss(x, x_recon, "mse")
        assert loss.item() >= 0
        assert ae.compute_loss(x, x, "mse").item() == pytest.approx(0.0)

    def test_bce_loss(self):
        """BCE loss should be non-negative."""
        from autoencoders_pytorch.autoencoder import Autoencoder
        ae = Autoencoder(784, 256, 32)
        x = torch.rand(16, 784)  # [0, 1] for BCE
        x_recon, _ = ae(x)
        loss = ae.compute_loss(x, x_recon, "bce")
        assert loss.item() >= 0

    def test_backward_pass(self):
        """All parameters should receive gradients."""
        from autoencoders_pytorch.autoencoder import Autoencoder
        ae = Autoencoder(784, 256, 32)
        x = torch.randn(16, 784)
        x_recon, z = ae(x)
        loss = ae.compute_loss(x, x_recon)
        loss.backward()
        for p in ae.parameters():
            assert p.grad is not None


class TestVAE:
    """Tests for Variational Autoencoder."""

    def test_encoder_outputs_mu_and_logvar(self):
        """VAE encoder should output mu and log(sigma^2)."""
        from autoencoders_pytorch.vae import VAEEncoder
        batch, d, hidden, k = 16, 784, 256, 20
        enc = VAEEncoder(d, hidden, k)
        x = torch.randn(batch, d)
        mu, logvar = enc(x)
        assert mu.shape == (batch, k)
        assert logvar.shape == (batch, k)

    def test_reparameterize_shape(self):
        """Reparameterization should output same shape as mu/logvar."""
        from autoencoders_pytorch.vae import reparameterize
        batch, k = 16, 20
        mu = torch.randn(batch, k)
        logvar = torch.randn(batch, k)
        z = reparameterize(mu, logvar)
        assert z.shape == (batch, k)

    def test_kl_divergence_non_negative(self):
        """KL divergence should be non-negative."""
        from autoencoders_pytorch.vae import kl_divergence
        mu = torch.randn(16, 20)
        logvar = torch.randn(16, 20)
        kl = kl_divergence(mu, logvar)
        assert kl.item() >= 0

    def test_kl_divergence_zero_at_prior(self):
        """KL should be ~0 when q(z|x) = p(z) = N(0, I)."""
        from autoencoders_pytorch.vae import kl_divergence
        mu = torch.zeros(16, 20)
        logvar = torch.zeros(16, 20)
        kl = kl_divergence(mu, logvar)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_vae_forward_shapes(self):
        """VAE forward should return correct shapes."""
        from autoencoders_pytorch.vae import VAE
        batch, d, hidden, k = 16, 784, 256, 20
        vae = VAE(d, hidden, k)
        x = torch.randn(batch, d)
        x_recon, mu, logvar, z = vae(x)
        assert x_recon.shape == (batch, d)
        assert mu.shape == (batch, k)
        assert logvar.shape == (batch, k)
        assert z.shape == (batch, k)

    def test_vae_sample_shape(self):
        """VAE sample should generate correct shape."""
        from autoencoders_pytorch.vae import VAE
        d, hidden, k = 784, 256, 20
        vae = VAE(d, hidden, k)
        samples = vae.sample(10)
        assert samples.shape == (10, d)

    def test_vae_interpolate_shape(self):
        """VAE interpolate should return n_steps samples."""
        from autoencoders_pytorch.vae import VAE
        d, hidden, k = 784, 256, 20
        vae = VAE(d, hidden, k)
        x1 = torch.randn(1, d)
        x2 = torch.randn(1, d)
        interp = vae.interpolate(x1, x2, n_steps=10)
        assert interp.shape == (10, d)

    def test_vae_backward(self):
        """All VAE parameters should receive gradients."""
        from autoencoders_pytorch.vae import VAE
        vae = VAE(784, 256, 20)
        x = torch.randn(16, 784)
        x_recon, mu, logvar, z = vae(x)
        total, _, _ = vae.compute_loss(x, x_recon, mu, logvar)
        total.backward()
        for p in vae.parameters():
            assert p.grad is not None


class TestBetaVAE:
    """Tests for Beta-VAE."""

    def test_loss_with_beta(self):
        """beta-VAE loss should weight KL by beta."""
        from autoencoders_pytorch.beta_vae import BetaVAE
        d, hidden, k = 784, 256, 10
        vae_b1 = BetaVAE(d, hidden, k, beta=1.0)
        vae_b4 = BetaVAE(d, hidden, k, beta=4.0)

        # Use same weights for fair comparison
        vae_b4.load_state_dict(vae_b1.state_dict())

        x = torch.randn(16, d)
        x_recon1, mu1, logvar1, _ = vae_b1(x)
        x_recon4, mu4, logvar4, _ = vae_b4(x)

        _, _, kl1, wkl1 = vae_b1.compute_loss(x, x_recon1, mu1, logvar1)
        _, _, kl4, wkl4 = vae_b4.compute_loss(x, x_recon4, mu4, logvar4)

        # Same weights => same raw KL, weighted KL differs by factor of 4
        assert kl1.item() == pytest.approx(kl4.item(), rel=1e-4)
        assert wkl4.item() == pytest.approx(4.0 * wkl1.item(), rel=1e-4)

    def test_traverse_latent_shape(self):
        """Latent traversal should return n_steps samples."""
        from autoencoders_pytorch.beta_vae import BetaVAE
        d, hidden, k = 784, 256, 10
        vae = BetaVAE(d, hidden, k, beta=4.0)
        x = torch.randn(1, d)
        traversals = vae.traverse_latent(x, dim=0, n_steps=10)
        assert traversals.shape == (10, d)

    def test_annealed_beta_warmup(self):
        """Annealed beta should increase during warmup."""
        from autoencoders_pytorch.beta_vae import AnnealedBetaVAE
        vae = AnnealedBetaVAE(784, 256, 10, beta_max=4.0, warmup_steps=100)

        vae.step = 0
        beta_start = vae.get_beta()
        vae.step = 50
        beta_mid = vae.get_beta()
        vae.step = 100
        beta_end = vae.get_beta()

        assert beta_start < beta_mid < beta_end
        assert beta_end == pytest.approx(4.0)


class TestCVAE:
    """Tests for Conditional VAE."""

    def test_one_hot_encode(self):
        """One-hot encoding should work correctly."""
        from autoencoders_pytorch.cvae import one_hot_encode
        labels = torch.tensor([0, 1, 2, 9])
        one_hot = one_hot_encode(labels, 10)
        assert one_hot.shape == (4, 10)
        assert torch.all(one_hot.sum(dim=1) == 1)
        assert one_hot[0, 0] == 1
        assert one_hot[3, 9] == 1

    def test_encoder_with_condition(self):
        """CVAE encoder should accept condition."""
        from autoencoders_pytorch.cvae import CVAEEncoder
        batch, d, c_dim, hidden, k = 16, 784, 10, 256, 20
        enc = CVAEEncoder(d, c_dim, hidden, k)
        x = torch.randn(batch, d)
        c = torch.randn(batch, c_dim)
        mu, logvar = enc(x, c)
        assert mu.shape == (batch, k)
        assert logvar.shape == (batch, k)

    def test_decoder_with_condition(self):
        """CVAE decoder should accept condition."""
        from autoencoders_pytorch.cvae import CVAEDecoder
        batch, d, c_dim, hidden, k = 16, 784, 10, 256, 20
        dec = CVAEDecoder(k, c_dim, hidden, d)
        z = torch.randn(batch, k)
        c = torch.randn(batch, c_dim)
        x_recon = dec(z, c)
        assert x_recon.shape == (batch, d)

    def test_cvae_forward_shapes(self):
        """CVAE forward should return correct shapes."""
        from autoencoders_pytorch.cvae import CVAE
        batch, d, c_dim, hidden, k = 16, 784, 10, 256, 20
        cvae = CVAE(d, c_dim, hidden, k)
        x = torch.randn(batch, d)
        c = torch.randn(batch, c_dim)
        x_recon, mu, logvar, z = cvae(x, c)
        assert x_recon.shape == (batch, d)
        assert mu.shape == (batch, k)

    def test_cvae_sample_class(self):
        """CVAE should generate samples for specific class."""
        from autoencoders_pytorch.cvae import CVAE
        d, c_dim, hidden, k = 784, 10, 256, 20
        cvae = CVAE(d, c_dim, hidden, k)
        samples = cvae.sample_class(5, 10, n_samples=8)
        assert samples.shape == (8, d)

    def test_cvae_interpolate_class(self):
        """CVAE should interpolate between conditions."""
        from autoencoders_pytorch.cvae import CVAE, one_hot_encode
        d, c_dim, hidden, k = 784, 10, 256, 20
        cvae = CVAE(d, c_dim, hidden, k)
        x = torch.randn(1, d)
        c1 = one_hot_encode(torch.tensor([0]), 10)
        c2 = one_hot_encode(torch.tensor([9]), 10)
        interp = cvae.interpolate_class(x, c1, c2, n_steps=10)
        assert interp.shape == (10, d)

    def test_cvae_backward(self):
        """All CVAE parameters should receive gradients."""
        from autoencoders_pytorch.cvae import CVAE
        cvae = CVAE(784, 10, 256, 20)
        x = torch.randn(16, 784)
        c = torch.randn(16, 10)
        x_recon, mu, logvar, z = cvae(x, c)
        total, _, _ = cvae.compute_loss(x, x_recon, mu, logvar)
        total.backward()
        for p in cvae.parameters():
            assert p.grad is not None


class TestVQVAE:
    """Tests for VQ-VAE."""

    def test_quantizer_output_shapes(self):
        """Quantizer should return correct shapes."""
        from autoencoders_pytorch.vqvae import VectorQuantizer
        batch, k, n_emb = 16, 64, 512
        vq = VectorQuantizer(n_emb, k)
        z_e = torch.randn(batch, k)
        z_q, indices, vq_loss = vq(z_e)
        assert z_q.shape == (batch, k)
        assert indices.shape == (batch,)
        assert vq_loss.item() >= 0

    def test_indices_valid(self):
        """Indices should be valid codebook indices."""
        from autoencoders_pytorch.vqvae import VectorQuantizer
        vq = VectorQuantizer(512, 64)
        z_e = torch.randn(32, 64)
        _, indices, _ = vq(z_e)
        assert torch.all(indices >= 0)
        assert torch.all(indices < 512)

    def test_vq_loss_components(self):
        """VQ loss should be non-negative."""
        from autoencoders_pytorch.vqvae import VectorQuantizer
        vq = VectorQuantizer(512, 64, beta=0.25)
        z_e = torch.randn(16, 64)
        _, _, vq_loss = vq(z_e)
        assert vq_loss.item() >= 0

    def test_vqvae_forward_shapes(self):
        """VQ-VAE forward should return correct shapes."""
        from autoencoders_pytorch.vqvae import VQVAE
        batch, d, hidden, k, n_emb = 16, 784, 256, 64, 512
        vqvae = VQVAE(d, hidden, k, n_emb)
        x = torch.randn(batch, d)
        x_recon, z_e, z_q, indices, vq_loss = vqvae(x)
        assert x_recon.shape == (batch, d)
        assert z_e.shape == (batch, k)
        assert z_q.shape == (batch, k)
        assert indices.shape == (batch,)

    def test_vqvae_encode_discrete(self):
        """VQ-VAE encode should return discrete indices."""
        from autoencoders_pytorch.vqvae import VQVAE
        batch, d = 16, 784
        vqvae = VQVAE(d, 256, 64, 512)
        x = torch.randn(batch, d)
        indices = vqvae.encode(x)
        assert indices.dtype in [torch.int32, torch.int64]
        assert indices.shape == (batch,)

    def test_vqvae_decode_from_indices(self):
        """Should decode from discrete indices."""
        from autoencoders_pytorch.vqvae import VQVAE
        batch, d = 16, 784
        vqvae = VQVAE(d, 256, 64, 512)
        indices = torch.randint(0, 512, (batch,))
        x_recon = vqvae.decode_from_indices(indices)
        assert x_recon.shape == (batch, d)

    def test_codebook_usage_perplexity(self):
        """Codebook usage should compute perplexity."""
        from autoencoders_pytorch.vqvae import VQVAE
        vqvae = VQVAE(784, 256, 64, 512)
        data = torch.randn(100, 784)
        usage, perplexity = vqvae.get_codebook_usage(data)
        assert usage.shape == (512,)
        assert perplexity.item() > 0
        assert perplexity.item() <= 512

    def test_vqvae_backward(self):
        """Straight-through estimator should allow gradients to flow."""
        from autoencoders_pytorch.vqvae import VQVAE
        vqvae = VQVAE(784, 256, 64, 512)
        x = torch.randn(16, 784)
        x_recon, z_e, z_q, indices, vq_loss = vqvae(x)
        total, _, _ = vqvae.compute_loss(x, x_recon, vq_loss)
        total.backward()
        # Encoder should get gradients through straight-through estimator
        for p in vqvae.encoder.parameters():
            assert p.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])