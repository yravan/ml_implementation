import numpy as np
import pytest
from rl.dagger import (
    behavior_cloning_loss, behavior_cloning_loss_discrete,
    aggregate_dataset, dagger_label, dagger_loss,
)


class TestBehaviorCloningLoss:
    """Tests for continuous behavior cloning loss (MSE)."""

    def test_perfect_match(self):
        """Zero loss when predictions match expert."""
        actions = np.array([[1.0, 2.0], [3.0, 4.0]])
        loss, grad = behavior_cloning_loss(actions, actions)
        np.testing.assert_almost_equal(loss, 0.0)
        np.testing.assert_array_almost_equal(grad, np.zeros_like(actions))

    def test_known_loss(self):
        """Verify against hand-computed MSE."""
        pred = np.array([[0.0], [0.0]])
        expert = np.array([[2.0], [4.0]])
        loss, _ = behavior_cloning_loss(pred, expert)
        # MSE = mean(4 + 16) = 10
        np.testing.assert_almost_equal(loss, 10.0)

    def test_gradient_shape(self):
        pred = np.random.randn(5, 3)
        expert = np.random.randn(5, 3)
        _, grad = behavior_cloning_loss(pred, expert)
        assert grad.shape == pred.shape

    def test_gradient_direction(self):
        """Gradient should point from prediction toward expert."""
        pred = np.array([[0.0]])
        expert = np.array([[1.0]])
        _, grad = behavior_cloning_loss(pred, expert)
        # d/d_pred MSE = 2(pred - expert)/N -> negative (pushes pred toward expert)
        assert grad[0, 0] < 0

    def test_1d_actions(self):
        """Should work with 1D action arrays."""
        pred = np.array([1.0, 2.0, 3.0])
        expert = np.array([1.0, 2.0, 3.0])
        loss, grad = behavior_cloning_loss(pred, expert)
        np.testing.assert_almost_equal(loss, 0.0)


class TestBehaviorCloningLossDiscrete:
    """Tests for discrete behavior cloning loss (cross-entropy)."""

    def test_shape(self):
        logits = np.random.randn(5, 3)
        actions = np.array([0, 1, 2, 0, 1])
        loss, grad = behavior_cloning_loss_discrete(logits, actions)
        assert np.isscalar(loss) or loss.ndim == 0
        assert grad.shape == logits.shape

    def test_low_loss_for_correct_prediction(self):
        """High logit for correct action should give low loss."""
        logits = np.array([[10.0, -10.0, -10.0]])  # Strong prediction for action 0
        actions = np.array([0])
        loss, _ = behavior_cloning_loss_discrete(logits, actions)
        assert loss < 0.01

    def test_high_loss_for_wrong_prediction(self):
        """High logit for wrong action should give high loss."""
        logits = np.array([[-10.0, 10.0, -10.0]])  # Strong prediction for action 1
        actions = np.array([0])  # But expert says action 0
        loss, _ = behavior_cloning_loss_discrete(logits, actions)
        assert loss > 10.0

    def test_non_negative(self):
        np.random.seed(42)
        logits = np.random.randn(10, 4)
        actions = np.random.randint(0, 4, size=10)
        loss, _ = behavior_cloning_loss_discrete(logits, actions)
        assert loss >= 0


class TestAggregateDataset:
    """Tests for dataset aggregation."""

    def test_sizes_add(self):
        s_old = np.random.randn(10, 3)
        a_old = np.random.randn(10, 2)
        s_new = np.random.randn(5, 3)
        a_new = np.random.randn(5, 2)
        s_agg, a_agg = aggregate_dataset(s_old, a_old, s_new, a_new)
        assert s_agg.shape == (15, 3)
        assert a_agg.shape == (15, 2)

    def test_contains_old_data(self):
        """Aggregated set should contain all old data."""
        s_old = np.array([[1, 2], [3, 4]])
        a_old = np.array([[10], [20]])
        s_new = np.array([[5, 6]])
        a_new = np.array([[30]])
        s_agg, a_agg = aggregate_dataset(s_old, a_old, s_new, a_new)
        # Old states should appear in aggregated (order may vary)
        assert np.any(np.all(s_agg == [1, 2], axis=1))
        assert np.any(np.all(s_agg == [3, 4], axis=1))

    def test_contains_new_data(self):
        """Aggregated set should contain all new data."""
        s_old = np.array([[1, 2]])
        a_old = np.array([[10]])
        s_new = np.array([[5, 6], [7, 8]])
        a_new = np.array([[30], [40]])
        s_agg, _ = aggregate_dataset(s_old, a_old, s_new, a_new)
        assert np.any(np.all(s_agg == [5, 6], axis=1))
        assert np.any(np.all(s_agg == [7, 8], axis=1))

    def test_1d_actions(self):
        """Should work with 1D action arrays."""
        s_old = np.random.randn(3, 2)
        a_old = np.array([0, 1, 2])
        s_new = np.random.randn(2, 2)
        a_new = np.array([1, 0])
        s_agg, a_agg = aggregate_dataset(s_old, a_old, s_new, a_new)
        assert a_agg.shape == (5,)


class TestDaggerLabel:
    """Tests for expert labeling."""

    def test_calls_expert(self):
        """Should return expert's actions for given states."""
        def expert(states):
            return states[:, 0:1] * 2  # Simple expert: doubles first feature

        states = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        labels = dagger_label(expert, states)
        expected = np.array([[2.0], [4.0], [6.0]])
        np.testing.assert_array_almost_equal(labels, expected)

    def test_shape(self):
        def expert(states):
            return np.zeros((states.shape[0], 2))

        states = np.random.randn(10, 4)
        labels = dagger_label(expert, states)
        assert labels.shape == (10, 2)


class TestDaggerLoss:
    """Tests for full DAgger iteration loss."""

    def test_returns_aggregated_dataset(self):
        """Should return the aggregated dataset."""
        def learner(states):
            return np.zeros((states.shape[0], 1))

        def expert(states):
            return np.ones((states.shape[0], 1))

        rollout_states = np.random.randn(5, 2)
        s_old = np.random.randn(10, 2)
        a_old = np.ones((10, 1))

        loss, s_agg, a_agg = dagger_loss(learner, expert, rollout_states, s_old, a_old)
        assert s_agg.shape == (15, 2)
        assert a_agg.shape == (15, 1)

    def test_loss_is_scalar(self):
        def learner(states):
            return np.zeros((states.shape[0], 1))

        def expert(states):
            return np.ones((states.shape[0], 1))

        rollout_states = np.random.randn(5, 2)
        s_old = np.random.randn(3, 2)
        a_old = np.ones((3, 1))

        loss, _, _ = dagger_loss(learner, expert, rollout_states, s_old, a_old)
        assert np.isscalar(loss) or (hasattr(loss, 'ndim') and loss.ndim == 0)

    def test_perfect_learner_zero_loss(self):
        """If learner matches expert, loss should be zero."""
        def policy(states):
            return states[:, :1]  # Same function

        rollout_states = np.random.randn(5, 2)
        s_old = np.random.randn(3, 2)
        a_old = s_old[:, :1]  # Expert labels match learner

        loss, _, _ = dagger_loss(policy, policy, rollout_states, s_old, a_old)
        np.testing.assert_almost_equal(loss, 0.0, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
