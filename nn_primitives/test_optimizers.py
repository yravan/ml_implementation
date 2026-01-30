import numpy as np
import pytest
from nn_primitives.optimizers import sgd_step, sgd_momentum_step, adam_step


class TestSGD:
    """Tests for vanilla SGD."""

    def test_basic_update(self):
        params = [np.array([1.0, 2.0])]
        grads = [np.array([0.1, 0.2])]
        params = sgd_step(params, grads, lr=1.0)
        np.testing.assert_array_almost_equal(params[0], [0.9, 1.8])

    def test_zero_gradient(self):
        """Zero gradient should not change params."""
        params = [np.array([5.0, 6.0])]
        grads = [np.zeros(2)]
        params = sgd_step(params, grads, lr=0.1)
        np.testing.assert_array_almost_equal(params[0], [5.0, 6.0])

    def test_learning_rate_scaling(self):
        """Larger lr should make bigger steps."""
        params1 = [np.array([1.0])]
        params2 = [np.array([1.0])]
        grads = [np.array([1.0])]
        p1 = sgd_step(params1, grads, lr=0.1)
        p2 = sgd_step(params2, grads, lr=0.5)
        assert abs(p2[0][0] - 1.0) > abs(p1[0][0] - 1.0)

    def test_multiple_params(self):
        params = [np.array([1.0]), np.array([2.0, 3.0])]
        grads = [np.array([0.5]), np.array([1.0, 1.0])]
        params = sgd_step(params, grads, lr=0.1)
        np.testing.assert_almost_equal(params[0][0], 0.95)
        np.testing.assert_array_almost_equal(params[1], [1.9, 2.9])


class TestSGDMomentum:
    """Tests for SGD with momentum."""

    def test_first_step_like_sgd(self):
        """First step with zero velocity should be like vanilla SGD."""
        params = [np.array([1.0, 2.0])]
        grads = [np.array([0.1, 0.2])]
        velocities = [np.zeros(2)]
        params, _ = sgd_momentum_step(params, grads, velocities, lr=1.0, momentum=0.9)
        np.testing.assert_array_almost_equal(params[0], [0.9, 1.8])

    def test_velocity_accumulation(self):
        """Repeated same gradient should accelerate via momentum."""
        params = [np.array([0.0])]
        velocities = [np.zeros(1)]
        grads = [np.array([1.0])]

        steps = []
        for _ in range(5):
            old_p = params[0].copy()
            params, velocities = sgd_momentum_step(params, grads, velocities, lr=0.1, momentum=0.9)
            steps.append(abs(params[0][0] - old_p[0]))

        # Steps should increase due to momentum
        assert steps[-1] > steps[0]

    def test_velocity_returned(self):
        params = [np.array([1.0])]
        velocities = [np.zeros(1)]
        grads = [np.array([1.0])]
        _, vel = sgd_momentum_step(params, grads, velocities, lr=0.1, momentum=0.9)
        assert len(vel) == 1
        assert vel[0].shape == (1,)


class TestAdam:
    """Tests for Adam optimizer."""

    def test_basic_update(self):
        """Params should change after one step."""
        params = [np.array([1.0, 2.0])]
        grads = [np.array([0.1, 0.2])]
        m = [np.zeros(2)]
        v = [np.zeros(2)]
        params, m, v = adam_step(params, grads, m, v, t=1, lr=0.001)
        assert not np.allclose(params[0], [1.0, 2.0])

    def test_zero_gradient(self):
        """Zero gradient should not change params."""
        params = [np.array([5.0])]
        grads = [np.array([0.0])]
        m = [np.zeros(1)]
        v = [np.zeros(1)]
        params, _, _ = adam_step(params, grads, m, v, t=1, lr=0.001)
        np.testing.assert_almost_equal(params[0][0], 5.0)

    def test_bias_correction(self):
        """Early steps should apply bias correction (larger effective lr)."""
        params = [np.array([0.0])]
        grads = [np.array([1.0])]
        m = [np.zeros(1)]
        v = [np.zeros(1)]
        params, m, v = adam_step(params, grads, m, v, t=1, lr=0.1)
        # With bias correction at t=1: m_hat = 0.1 / (1-0.9) = 1.0
        # v_hat = 0.001 / (1-0.999) = 1.0
        # update = 0.1 * 1.0 / (1.0 + 1e-8) ≈ 0.1
        np.testing.assert_almost_equal(params[0][0], -0.1, decimal=2)

    def test_moments_updated(self):
        """First and second moments should be updated."""
        params = [np.array([0.0])]
        grads = [np.array([2.0])]
        m = [np.zeros(1)]
        v = [np.zeros(1)]
        _, m_new, v_new = adam_step(params, grads, m, v, t=1)
        assert m_new[0][0] > 0  # First moment should be positive
        assert v_new[0][0] > 0  # Second moment should be positive

    def test_multiple_params(self):
        params = [np.array([1.0]), np.array([2.0, 3.0])]
        grads = [np.array([0.5]), np.array([1.0, -1.0])]
        m = [np.zeros(1), np.zeros(2)]
        v = [np.zeros(1), np.zeros(2)]
        params, m, v = adam_step(params, grads, m, v, t=1, lr=0.01)
        assert len(params) == 2
        assert len(m) == 2
        assert len(v) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
