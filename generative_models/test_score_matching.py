import torch
import pytest
from generative_models.score_matching import (
    denoising_score_matching_loss,
    score_to_noise_pred, noise_pred_to_score,
    langevin_dynamics_step,
)


class TestDenoisingScoreMatchingLoss:
    """Tests for denoising score matching loss."""

    def test_zero_for_perfect(self):
        """If score_pred = -noise/sigma, loss should be 0."""
        noise = torch.randn(4, 3)
        sigma = 0.5
        score_pred = -noise / sigma
        loss, _ = denoising_score_matching_loss(score_pred, noise, sigma)
        assert abs(loss) < 1e-10

    def test_positive_for_imperfect(self):
        score_pred = torch.randn(4, 3)
        noise = torch.randn(4, 3)
        loss, _ = denoising_score_matching_loss(score_pred, noise, sigma=1.0)
        assert loss > 0

    def test_gradient_shape(self):
        score_pred = torch.randn(5, 2)
        noise = torch.randn(5, 2)
        _, grad = denoising_score_matching_loss(score_pred, noise, sigma=0.5)
        assert grad.shape == score_pred.shape

    def test_gradient_zero_at_optimum(self):
        """Gradient should be zero when score matches target."""
        noise = torch.randn(3, 4)
        sigma = 0.3
        score_pred = -noise / sigma
        _, grad = denoising_score_matching_loss(score_pred, noise, sigma)
        torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-10, rtol=0)


class TestScoreNoiseConversions:
    """Tests for converting between score and noise predictions."""

    def test_roundtrip_score_to_noise(self):
        """score -> noise -> score should recover original."""
        score = torch.randn(5, 3)
        sigma = 0.7
        noise = score_to_noise_pred(score, sigma)
        score_back = noise_pred_to_score(noise, sigma)
        torch.testing.assert_close(score_back, score)

    def test_roundtrip_noise_to_score(self):
        noise = torch.randn(5, 3)
        sigma = 1.5
        score = noise_pred_to_score(noise, sigma)
        noise_back = score_to_noise_pred(score, sigma)
        torch.testing.assert_close(noise_back, noise)

    def test_score_to_noise_formula(self):
        """noise = -sigma * score."""
        score = torch.tensor([[1.0, -2.0]])
        sigma = 0.5
        noise = score_to_noise_pred(score, sigma)
        torch.testing.assert_close(noise, torch.tensor([[-0.5, 1.0]]))

    def test_noise_to_score_formula(self):
        """score = -noise / sigma."""
        noise = torch.tensor([[2.0, -4.0]])
        sigma = 2.0
        score = noise_pred_to_score(noise, sigma)
        torch.testing.assert_close(score, torch.tensor([[-1.0, 2.0]]))


class TestLangevinDynamics:
    """Tests for Langevin dynamics sampling."""

    def test_shape(self):
        x = torch.randn(10, 3)
        score = torch.randn(10, 3)
        x_new = langevin_dynamics_step(x, score, step_size=0.01)
        assert x_new.shape == x.shape

    def test_moves_toward_mode(self):
        """Score pointing toward origin should move samples toward origin."""
        torch.manual_seed(42)
        x = torch.tensor([[10.0, 10.0]])  # Far from origin
        # Score = -x (pointing toward origin for Gaussian)
        score = -x
        x_new = langevin_dynamics_step(x, score, step_size=0.1)
        # Should be closer to origin
        assert torch.norm(x_new).item() < torch.norm(x).item()

    def test_zero_score_adds_noise(self):
        """With zero score, step just adds noise."""
        x = torch.zeros(100, 2)
        score = torch.zeros(100, 2)
        x_new = langevin_dynamics_step(x, score, step_size=1.0)
        # Should have non-zero mean squared distance from origin
        assert torch.mean(x_new**2).item() > 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
