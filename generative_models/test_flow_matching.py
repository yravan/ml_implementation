import torch
import math
import pytest
from generative_models.flow_matching import (
    conditional_ot_path, flow_matching_loss, euler_integrate,
)


class TestConditionalOTPath:
    """Tests for conditional optimal transport path."""

    def test_t0_is_source(self):
        """At t=0, x_t should be x_0."""
        x_0 = torch.randn(5, 3)
        x_1 = torch.randn(5, 3)
        x_t, _ = conditional_ot_path(x_0, x_1, t=0.0)
        torch.testing.assert_close(x_t, x_0)

    def test_t1_is_target(self):
        """At t=1, x_t should be x_1."""
        x_0 = torch.randn(5, 3)
        x_1 = torch.randn(5, 3)
        x_t, _ = conditional_ot_path(x_0, x_1, t=1.0)
        torch.testing.assert_close(x_t, x_1)

    def test_midpoint(self):
        """At t=0.5, x_t should be the midpoint."""
        x_0 = torch.tensor([[0.0, 0.0]])
        x_1 = torch.tensor([[2.0, 4.0]])
        x_t, _ = conditional_ot_path(x_0, x_1, t=0.5)
        torch.testing.assert_close(x_t, torch.tensor([[1.0, 2.0]]))

    def test_velocity_is_constant(self):
        """Target velocity should be x_1 - x_0 (constant)."""
        x_0 = torch.tensor([[1.0, 2.0]])
        x_1 = torch.tensor([[4.0, 6.0]])
        _, u_t = conditional_ot_path(x_0, x_1, t=0.3)
        torch.testing.assert_close(u_t, torch.tensor([[3.0, 4.0]]))

    def test_batch_different_t(self):
        """Should handle per-sample t values."""
        x_0 = torch.zeros(3, 2)
        x_1 = torch.ones(3, 2)
        t = torch.tensor([[0.0], [0.5], [1.0]])
        x_t, _ = conditional_ot_path(x_0, x_1, t)
        torch.testing.assert_close(x_t[0], torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(x_t[1], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(x_t[2], torch.tensor([1.0, 1.0]))

    def test_shapes(self):
        x_0 = torch.randn(8, 4)
        x_1 = torch.randn(8, 4)
        x_t, u_t = conditional_ot_path(x_0, x_1, t=0.5)
        assert x_t.shape == (8, 4)
        assert u_t.shape == (8, 4)


class TestFlowMatchingLoss:
    """Tests for flow matching training loss."""

    def test_zero_for_perfect(self):
        """If v_pred = x_1 - x_0, loss should be 0."""
        x_0 = torch.randn(5, 3)
        x_1 = torch.randn(5, 3)
        v_pred = x_1 - x_0
        loss, _ = flow_matching_loss(v_pred, x_0, x_1)
        assert abs(loss) < 1e-7

    def test_positive_for_imperfect(self):
        v_pred = torch.randn(5, 3)
        x_0 = torch.randn(5, 3)
        x_1 = torch.randn(5, 3)
        loss, _ = flow_matching_loss(v_pred, x_0, x_1)
        assert loss > 0

    def test_gradient_shape(self):
        v_pred = torch.randn(4, 2)
        x_0 = torch.randn(4, 2)
        x_1 = torch.randn(4, 2)
        _, grad = flow_matching_loss(v_pred, x_0, x_1)
        assert grad.shape == v_pred.shape

    def test_gradient_zero_at_optimum(self):
        x_0 = torch.randn(3, 2)
        x_1 = torch.randn(3, 2)
        v_pred = x_1 - x_0
        _, grad = flow_matching_loss(v_pred, x_0, x_1)
        torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-7, rtol=0)


class TestEulerIntegrate:
    """Tests for Euler integration of velocity field."""

    def test_identity_velocity(self):
        """Constant velocity v(x,t) = c should give x_final = x_init + c."""
        x_init = torch.zeros(3, 2)
        c = torch.tensor([1.0, 2.0])

        def v_fn(x, t):
            return c.unsqueeze(0).expand(x.shape[0], -1)

        x_final, traj = euler_integrate(x_init, v_fn, n_steps=100)
        expected = c.unsqueeze(0).expand(3, -1)
        torch.testing.assert_close(x_final, expected, atol=0.05, rtol=0)

    def test_trajectory_shape(self):
        x_init = torch.randn(5, 3)

        def v_fn(x, t):
            return torch.zeros_like(x)

        _, traj = euler_integrate(x_init, v_fn, n_steps=50)
        assert traj.shape == (51, 5, 3)  # n_steps + 1

    def test_trajectory_starts_at_init(self):
        x_init = torch.tensor([[1.0, 2.0]])

        def v_fn(x, t):
            return torch.zeros_like(x)

        _, traj = euler_integrate(x_init, v_fn, n_steps=10)
        torch.testing.assert_close(traj[0], x_init)

    def test_zero_velocity(self):
        """Zero velocity field: x_final = x_init."""
        x_init = torch.tensor([[3.0, 4.0]])

        def v_fn(x, t):
            return torch.zeros_like(x)

        x_final, _ = euler_integrate(x_init, v_fn, n_steps=100)
        torch.testing.assert_close(x_final, x_init)

    def test_more_steps_more_accurate(self):
        """More Euler steps should give better approximation."""
        x_init = torch.tensor([[1.0]])

        # Velocity v = x (exponential growth: x(1) = e)
        def v_fn(x, t):
            return x

        x_10, _ = euler_integrate(x_init, v_fn, n_steps=10)
        x_1000, _ = euler_integrate(x_init, v_fn, n_steps=1000)
        true_val = math.e
        err_10 = abs(x_10[0, 0].item() - true_val)
        err_1000 = abs(x_1000[0, 0].item() - true_val)
        assert err_1000 < err_10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
