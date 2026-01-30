import numpy as np
import pytest
from rl.policy_gradient import (
    discounted_returns, softmax_policy, log_softmax_policy,
    reinforce_loss, reinforce_gradient, gae,
)


class TestDiscountedReturns:
    """Tests for computing discounted returns G_t."""

    def test_single_reward(self):
        """Single reward: G_0 = r_0."""
        returns = discounted_returns(np.array([5.0]), 0.9)
        np.testing.assert_array_almost_equal(returns, [5.0])

    def test_no_discount(self):
        """With gamma=1, G_t = sum of all future rewards."""
        rewards = np.array([1.0, 2.0, 3.0])
        returns = discounted_returns(rewards, 1.0)
        np.testing.assert_array_almost_equal(returns, [6.0, 5.0, 3.0])

    def test_zero_discount(self):
        """With gamma=0, G_t = r_t."""
        rewards = np.array([1.0, 2.0, 3.0])
        returns = discounted_returns(rewards, 0.0)
        np.testing.assert_array_almost_equal(returns, [1.0, 2.0, 3.0])

    def test_standard_case(self):
        """Standard discounting, gamma=0.5."""
        rewards = np.array([1.0, 2.0, 4.0])
        returns = discounted_returns(rewards, 0.5)
        # G_2 = 4
        # G_1 = 2 + 0.5 * 4 = 4
        # G_0 = 1 + 0.5 * 4 = 3
        np.testing.assert_array_almost_equal(returns, [3.0, 4.0, 4.0])

    def test_shape(self):
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = discounted_returns(rewards, 0.99)
        assert returns.shape == rewards.shape

    def test_monotone_last(self):
        """G_{T-1} always equals r_{T-1}."""
        rewards = np.array([10.0, 20.0, 30.0])
        returns = discounted_returns(rewards, 0.9)
        np.testing.assert_almost_equal(returns[-1], 30.0)

    def test_all_zeros(self):
        returns = discounted_returns(np.zeros(5), 0.9)
        np.testing.assert_array_almost_equal(returns, np.zeros(5))


class TestSoftmaxPolicy:
    """Tests for softmax policy parameterization."""

    def test_sums_to_one(self):
        """Probabilities should sum to 1."""
        theta = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])  # 3 actions, 2 features
        state = np.array([1.0, 0.0])
        probs = softmax_policy(theta, state)
        np.testing.assert_almost_equal(probs.sum(), 1.0)

    def test_non_negative(self):
        """All probabilities should be non-negative."""
        theta = np.random.randn(4, 3)
        state = np.random.randn(3)
        probs = softmax_policy(theta, state)
        assert np.all(probs >= 0)

    def test_shape(self):
        theta = np.random.randn(5, 3)
        state = np.random.randn(3)
        probs = softmax_policy(theta, state)
        assert probs.shape == (5,)

    def test_uniform_with_equal_logits(self):
        """Equal logits should give uniform distribution."""
        theta = np.ones((3, 2))
        state = np.array([1.0, 1.0])
        probs = softmax_policy(theta, state)
        np.testing.assert_array_almost_equal(probs, [1/3, 1/3, 1/3])

    def test_peaked_for_large_theta(self):
        """Large theta should make distribution peaked."""
        theta = np.array([[100.0], [0.0], [0.0]])
        state = np.array([1.0])
        probs = softmax_policy(theta, state)
        assert probs[0] > 0.99


class TestLogSoftmaxPolicy:
    """Tests for log-softmax policy."""

    def test_consistency_with_softmax(self):
        """exp(log_softmax) should equal softmax."""
        theta = np.random.randn(4, 3)
        state = np.random.randn(3)
        log_probs = log_softmax_policy(theta, state)
        probs = softmax_policy(theta, state)
        np.testing.assert_array_almost_equal(np.exp(log_probs), probs)

    def test_all_negative(self):
        """Log probabilities should be <= 0."""
        theta = np.random.randn(3, 2)
        state = np.random.randn(2)
        log_probs = log_softmax_policy(theta, state)
        assert np.all(log_probs <= 1e-10)

    def test_logsumexp_equals_zero(self):
        """log(sum(exp(log_probs))) should be 0 (probabilities sum to 1)."""
        theta = np.random.randn(5, 3)
        state = np.random.randn(3)
        log_probs = log_softmax_policy(theta, state)
        log_sum = np.log(np.sum(np.exp(log_probs)))
        np.testing.assert_almost_equal(log_sum, 0.0, decimal=5)


class TestReinforceLoss:
    """Tests for REINFORCE loss."""

    def test_positive_loss_for_good_actions(self):
        """Negative log_probs * positive returns -> positive loss."""
        log_probs = np.array([-1.0, -0.5, -2.0])
        returns = np.array([1.0, 2.0, 3.0])
        loss = reinforce_loss(log_probs, returns)
        assert loss > 0

    def test_zero_returns(self):
        """Zero returns should give zero loss."""
        log_probs = np.array([-1.0, -0.5])
        returns = np.zeros(2)
        loss = reinforce_loss(log_probs, returns)
        np.testing.assert_almost_equal(loss, 0.0)

    def test_scale_with_returns(self):
        """Doubling returns should double the loss magnitude."""
        log_probs = np.array([-1.0, -2.0])
        returns1 = np.array([1.0, 1.0])
        returns2 = np.array([2.0, 2.0])
        loss1 = reinforce_loss(log_probs, returns1)
        loss2 = reinforce_loss(log_probs, returns2)
        np.testing.assert_almost_equal(loss2, 2 * loss1)


class TestReinforceGradient:
    """Tests for REINFORCE policy gradient."""

    def test_shape(self):
        theta = np.zeros((3, 2))  # 3 actions, 2 features
        states = np.random.randn(5, 2)
        actions = np.array([0, 1, 2, 0, 1])
        returns = np.ones(5)
        grad = reinforce_gradient(theta, states, actions, returns)
        assert grad.shape == theta.shape

    def test_zero_returns_zero_gradient(self):
        """Zero returns should give zero gradient."""
        theta = np.random.randn(3, 2)
        states = np.random.randn(5, 2)
        actions = np.array([0, 1, 2, 0, 1])
        returns = np.zeros(5)
        grad = reinforce_gradient(theta, states, actions, returns)
        np.testing.assert_array_almost_equal(grad, np.zeros_like(theta))

    def test_gradient_direction(self):
        """Gradient should increase probability of high-return actions."""
        theta = np.zeros((2, 1))  # 2 actions, 1 feature
        states = np.array([[1.0]])  # 1 timestep
        actions = np.array([0])  # Action 0 taken
        returns = np.array([1.0])  # Positive return
        grad = reinforce_gradient(theta, states, actions, returns)
        # Gradient for action 0 should be positive (increase its prob)
        assert grad[0, 0] > 0


class TestGAE:
    """Tests for Generalized Advantage Estimation."""

    def test_shape(self):
        T = 5
        rewards = np.ones(T)
        values = np.zeros(T + 1)
        advantages = gae(rewards, values, 0.99, 0.95)
        assert advantages.shape == (T,)

    def test_lambda_zero_is_td_error(self):
        """λ=0: GAE reduces to the one-step TD error δ_t."""
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([0.5, 1.0, 1.5, 0.0])
        gamma = 0.9
        advantages = gae(rewards, values, gamma, lam=0.0)
        # A_t = δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        expected = np.array([
            1.0 + 0.9 * 1.0 - 0.5,   # 1.4
            2.0 + 0.9 * 1.5 - 1.0,   # 2.35
            3.0 + 0.9 * 0.0 - 1.5,   # 1.5
        ])
        np.testing.assert_array_almost_equal(advantages, expected)

    def test_lambda_one_is_mc_advantage(self):
        """λ=1: GAE reduces to Monte Carlo advantage (G_t - V(s_t))."""
        rewards = np.array([1.0, 1.0, 1.0])
        values = np.array([2.0, 2.0, 2.0, 0.0])
        gamma = 1.0
        advantages = gae(rewards, values, gamma, lam=1.0)
        # With gamma=1, lambda=1:
        # G_0 = 1+1+1 = 3, A_0 = 3 - 2 = 1
        # G_1 = 1+1 = 2, A_1 = 2 - 2 = 0
        # G_2 = 1, A_2 = 1 - 2 = -1
        np.testing.assert_array_almost_equal(advantages, [1.0, 0.0, -1.0])

    def test_terminal_bootstrap(self):
        """Terminal state (V(s_T) = 0) should be handled correctly."""
        rewards = np.array([1.0])
        values = np.array([0.5, 0.0])  # Terminal
        advantages = gae(rewards, values, 0.99, 0.95)
        # A_0 = δ_0 = 1 + 0.99*0 - 0.5 = 0.5
        np.testing.assert_array_almost_equal(advantages, [0.5])

    def test_zero_rewards_nonzero_values(self):
        """With zero rewards, advantage comes from value difference."""
        rewards = np.zeros(3)
        values = np.array([1.0, 0.9, 0.81, 0.729])
        gamma = 0.9
        advantages = gae(rewards, values, gamma, lam=0.0)
        # δ_t = 0 + γ V(s_{t+1}) - V(s_t)
        # δ_0 = 0.9*0.9 - 1.0 = -0.19
        expected_0 = gamma * values[1] - values[0]
        np.testing.assert_almost_equal(advantages[0], expected_0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
