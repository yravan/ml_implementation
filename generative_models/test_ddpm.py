import torch
import pytest
from generative_models.ddpm import (
    linear_noise_schedule, cosine_noise_schedule,
    forward_diffusion, predict_noise_loss, ddpm_sample_step,
)


class TestLinearNoiseSchedule:
    """Tests for linear noise schedule."""

    def test_shapes(self):
        betas, alphas, alpha_bars = linear_noise_schedule(100)
        assert betas.shape == (100,)
        assert alphas.shape == (100,)
        assert alpha_bars.shape == (100,)

    def test_beta_range(self):
        betas, _, _ = linear_noise_schedule(100, beta_start=1e-4, beta_end=0.02)
        torch.testing.assert_close(betas[0], torch.tensor(1e-4), atol=1e-5, rtol=0)
        torch.testing.assert_close(betas[-1], torch.tensor(0.02), atol=1e-5, rtol=0)

    def test_alpha_relationship(self):
        """alpha_t = 1 - beta_t."""
        betas, alphas, _ = linear_noise_schedule(50)
        torch.testing.assert_close(alphas, 1 - betas)

    def test_alpha_bar_is_cumprod(self):
        """alpha_bar_t = product of alpha_1 ... alpha_t."""
        _, alphas, alpha_bars = linear_noise_schedule(50)
        torch.testing.assert_close(alpha_bars, torch.cumprod(alphas, dim=0))

    def test_alpha_bar_monotone_decreasing(self):
        """alpha_bar should decrease over time (more noise)."""
        _, _, alpha_bars = linear_noise_schedule(100)
        assert torch.all(torch.diff(alpha_bars) < 0)

    def test_alpha_bar_range(self):
        """alpha_bar should be in (0, 1)."""
        _, _, alpha_bars = linear_noise_schedule(100)
        assert torch.all(alpha_bars > 0)
        assert torch.all(alpha_bars < 1)


class TestCosineNoiseSchedule:
    """Tests for cosine noise schedule."""

    def test_shapes(self):
        betas, alphas, alpha_bars = cosine_noise_schedule(100)
        assert betas.shape == (100,)
        assert alphas.shape == (100,)
        assert alpha_bars.shape == (100,)

    def test_alpha_bar_monotone(self):
        _, _, alpha_bars = cosine_noise_schedule(100)
        assert torch.all(torch.diff(alpha_bars) <= 0)

    def test_betas_clipped(self):
        """Betas should be in [0, 0.999]."""
        betas, _, _ = cosine_noise_schedule(100)
        assert torch.all(betas >= 0)
        assert torch.all(betas <= 0.999)

    def test_smoother_than_linear(self):
        """Cosine schedule should be smoother near t=0 than linear."""
        _, _, ab_cos = cosine_noise_schedule(100)
        _, _, ab_lin = linear_noise_schedule(100)
        # Cosine should start higher (less noise at early steps)
        assert ab_cos[0] > ab_lin[0]


class TestForwardDiffusion:
    """Tests for the forward diffusion process q(x_t | x_0)."""

    def test_shape(self):
        x_0 = torch.randn(8, 4)
        _, _, alpha_bars = linear_noise_schedule(100)
        t = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.long)
        x_t, noise = forward_diffusion(x_0, t, alpha_bars)
        assert x_t.shape == (8, 4)
        assert noise.shape == (8, 4)

    def test_t0_close_to_original(self):
        """At t=0 (alpha_bar ≈ 1), x_t ≈ x_0."""
        x_0 = torch.randn(5, 3)
        _, _, alpha_bars = linear_noise_schedule(1000)
        t = torch.zeros(5, dtype=torch.long)
        x_t, _ = forward_diffusion(x_0, t, alpha_bars)
        torch.testing.assert_close(x_t, x_0, atol=0.5, rtol=0)

    def test_large_t_mostly_noise(self):
        """At large t (alpha_bar ≈ 0), x_t should be mostly noise."""
        torch.manual_seed(42)
        x_0 = torch.ones(100, 3) * 10  # Strong signal
        _, _, alpha_bars = linear_noise_schedule(1000)
        t = torch.full((100,), 999, dtype=torch.long)
        x_t, _ = forward_diffusion(x_0, t, alpha_bars)
        # Mean of x_t should be close to 0 (noise dominates)
        assert abs(x_t.mean().item()) < 2

    def test_deterministic_with_fixed_noise(self):
        """Same noise should give same x_t."""
        x_0 = torch.randn(3, 2)
        _, _, alpha_bars = linear_noise_schedule(100)
        t = torch.tensor([50, 50, 50], dtype=torch.long)
        noise = torch.randn(3, 2)
        x_t1, _ = forward_diffusion(x_0, t, alpha_bars, noise=noise.clone())
        x_t2, _ = forward_diffusion(x_0, t, alpha_bars, noise=noise.clone())
        torch.testing.assert_close(x_t1, x_t2)

    def test_correct_formula(self):
        """x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise."""
        x_0 = torch.tensor([[3.0, 4.0]])
        _, _, alpha_bars = linear_noise_schedule(100)
        t = torch.tensor([50], dtype=torch.long)
        noise = torch.tensor([[1.0, -1.0]])
        x_t, _ = forward_diffusion(x_0, t, alpha_bars, noise=noise)
        ab = alpha_bars[50]
        expected = torch.sqrt(ab) * x_0 + torch.sqrt(1 - ab) * noise
        torch.testing.assert_close(x_t, expected)


class TestPredictNoiseLoss:
    """Tests for DDPM training loss."""

    def test_zero_for_perfect(self):
        noise = torch.randn(4, 3)
        loss, grad = predict_noise_loss(noise, noise)
        assert abs(loss) < 1e-7

    def test_positive(self):
        noise_pred = torch.randn(4, 3)
        noise_true = torch.randn(4, 3)
        loss, _ = predict_noise_loss(noise_pred, noise_true)
        assert loss > 0

    def test_gradient_shape(self):
        noise_pred = torch.randn(4, 3)
        noise_true = torch.randn(4, 3)
        _, grad = predict_noise_loss(noise_pred, noise_true)
        assert grad.shape == noise_pred.shape


class TestDDPMSampleStep:
    """Tests for DDPM reverse sampling step."""

    def test_shape(self):
        betas, alphas, alpha_bars = linear_noise_schedule(100)
        x_t = torch.randn(4, 3)
        noise_pred = torch.randn(4, 3)
        x_prev = ddpm_sample_step(x_t, t=50, noise_pred=noise_pred,
                                   betas=betas, alphas=alphas, alpha_bars=alpha_bars)
        assert x_prev.shape == (4, 3)

    def test_denoises(self):
        """With correct noise prediction, sample should move toward x_0."""
        torch.manual_seed(42)
        betas, alphas, alpha_bars = linear_noise_schedule(100)
        x_0 = torch.tensor([[5.0, 5.0]])
        t_idx = 10
        noise = torch.randn(1, 2)
        x_t, _ = forward_diffusion(x_0, torch.tensor([t_idx], dtype=torch.long),
                                    alpha_bars, noise=noise)
        # If we give the true noise as prediction, the mean should move toward x_0
        x_prev = ddpm_sample_step(x_t, t=t_idx, noise_pred=noise,
                                   betas=betas, alphas=alphas, alpha_bars=alpha_bars,
                                   z=torch.zeros(1, 2))
        # x_prev should be closer to x_0 than x_t
        dist_before = torch.norm(x_t - x_0).item()
        dist_after = torch.norm(x_prev - x_0).item()
        assert dist_after < dist_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
