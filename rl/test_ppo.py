import numpy as np
import pytest
from rl.ppo import (
    compute_ratio, ppo_clipped_objective, ppo_loss,
    value_function_loss, entropy_bonus,
)


class TestComputeRatio:
    """Tests for importance sampling ratio."""

    def test_same_policy(self):
        """Same log probs -> ratio = 1."""
        log_probs = np.array([-1.0, -2.0, -0.5])
        ratio = compute_ratio(log_probs, log_probs)
        np.testing.assert_array_almost_equal(ratio, np.ones(3))

    def test_shape(self):
        ratio = compute_ratio(np.zeros(5), np.zeros(5))
        assert ratio.shape == (5,)

    def test_known_ratio(self):
        """exp(new - old) for known values."""
        new_lp = np.array([-1.0])
        old_lp = np.array([-2.0])
        ratio = compute_ratio(new_lp, old_lp)
        # exp(-1 - (-2)) = exp(1) = e
        np.testing.assert_almost_equal(ratio[0], np.exp(1.0))

    def test_positive(self):
        """Ratios should always be positive."""
        np.random.seed(42)
        new_lp = -np.abs(np.random.randn(10))
        old_lp = -np.abs(np.random.randn(10))
        ratio = compute_ratio(new_lp, old_lp)
        assert np.all(ratio > 0)


class TestPPOClippedObjective:
    """Tests for PPO clipped surrogate objective."""

    def test_no_clipping_when_in_range(self):
        """When ratio is in [1-ε, 1+ε], L = r * A."""
        ratio = np.array([1.0, 1.05, 0.95])
        advantages = np.array([2.0, 3.0, -1.0])
        epsilon = 0.2
        obj = ppo_clipped_objective(ratio, advantages, epsilon)
        expected = ratio * advantages  # No clipping needed
        np.testing.assert_array_almost_equal(obj, expected)

    def test_clips_positive_advantage(self):
        """For A > 0, ratio is capped at 1+ε."""
        ratio = np.array([1.5])  # Above 1+ε
        advantages = np.array([2.0])
        epsilon = 0.2
        obj = ppo_clipped_objective(ratio, advantages, epsilon)
        # min(1.5 * 2, clip(1.5, 0.8, 1.2) * 2) = min(3.0, 1.2*2) = min(3, 2.4) = 2.4
        np.testing.assert_almost_equal(obj[0], 2.4)

    def test_clips_negative_advantage(self):
        """For A < 0, ratio is capped at 1-ε."""
        ratio = np.array([0.5])  # Below 1-ε
        advantages = np.array([-2.0])
        epsilon = 0.2
        obj = ppo_clipped_objective(ratio, advantages, epsilon)
        # min(0.5 * -2, clip(0.5, 0.8, 1.2) * -2) = min(-1, 0.8*-2) = min(-1, -1.6) = -1.6
        np.testing.assert_almost_equal(obj[0], -1.6)

    def test_shape(self):
        ratio = np.ones(10)
        advantages = np.zeros(10)
        obj = ppo_clipped_objective(ratio, advantages, 0.2)
        assert obj.shape == (10,)

    def test_zero_advantage(self):
        """Zero advantage -> zero objective regardless of ratio."""
        ratio = np.array([0.5, 1.0, 2.0])
        advantages = np.zeros(3)
        obj = ppo_clipped_objective(ratio, advantages, 0.2)
        np.testing.assert_array_almost_equal(obj, np.zeros(3))


class TestPPOLoss:
    """Tests for full PPO policy loss."""

    def test_zero_advantage_zero_loss(self):
        """Zero advantages should give zero loss."""
        loss = ppo_loss(np.zeros(5), np.zeros(5), np.zeros(5), 0.2)
        np.testing.assert_almost_equal(loss, 0.0)

    def test_positive_when_good_policy(self):
        """Loss should be negative of mean objective -> meaningful sign."""
        # Same policy -> ratio=1, objective = advantages
        log_probs = -np.ones(3)
        advantages = np.array([1.0, 2.0, 3.0])
        loss = ppo_loss(log_probs, log_probs, advantages, 0.2)
        # Loss = -mean(1*A) = -mean(1,2,3) = -2.0
        np.testing.assert_almost_equal(loss, -2.0)

    def test_scalar_output(self):
        loss = ppo_loss(np.zeros(5), np.zeros(5), np.ones(5), 0.2)
        assert np.isscalar(loss) or loss.ndim == 0


class TestValueFunctionLoss:
    """Tests for value function MSE loss."""

    def test_perfect_prediction(self):
        values = np.array([1.0, 2.0, 3.0])
        loss = value_function_loss(values, values)
        np.testing.assert_almost_equal(loss, 0.0)

    def test_known_mse(self):
        predicted = np.array([0.0, 0.0])
        targets = np.array([2.0, 4.0])
        loss = value_function_loss(predicted, targets)
        # MSE = mean(4 + 16) = 10
        np.testing.assert_almost_equal(loss, 10.0)

    def test_symmetric(self):
        """MSE is symmetric."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        np.testing.assert_almost_equal(
            value_function_loss(a, b),
            value_function_loss(b, a),
        )

    def test_non_negative(self):
        np.random.seed(42)
        for _ in range(10):
            pred = np.random.randn(10)
            target = np.random.randn(10)
            assert value_function_loss(pred, target) >= 0


class TestEntropyBonus:
    """Tests for policy entropy."""

    def test_uniform_max_entropy(self):
        """Uniform distribution has max entropy = log(n)."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        H = entropy_bonus(probs)
        np.testing.assert_almost_equal(H, np.log(4))

    def test_deterministic_zero_entropy(self):
        """Deterministic policy has zero entropy."""
        probs = np.array([1.0, 0.0, 0.0])
        H = entropy_bonus(probs)
        np.testing.assert_almost_equal(H, 0.0, decimal=5)

    def test_non_negative(self):
        np.random.seed(42)
        for _ in range(10):
            logits = np.random.randn(5)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            assert entropy_bonus(probs) >= -1e-10

    def test_binary(self):
        """Binary entropy: H(0.5, 0.5) = log(2)."""
        probs = np.array([0.5, 0.5])
        H = entropy_bonus(probs)
        np.testing.assert_almost_equal(H, np.log(2))

    def test_batch(self):
        """Should handle (T, n_actions) input."""
        probs = np.array([
            [0.5, 0.5],
            [1.0, 0.0],
            [0.25, 0.75],
        ])
        H = entropy_bonus(probs)
        assert H.shape == (3,)
        np.testing.assert_almost_equal(H[0], np.log(2))
        np.testing.assert_almost_equal(H[1], 0.0, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
