import numpy as np
import pytest
from robotics.jacobians import (
    planar_2r_jacobian, planar_3r_jacobian, numerical_jacobian,
    manipulability, is_singular, condition_number,
    pseudoinverse_ik_step, damped_pseudoinverse_ik_step,
    null_space_projection,
)
from robotics.forward_kinematics import planar_2r_fk, planar_3r_fk


class TestPlanar2RJacobian:
    """Tests for 2R planar arm Jacobian."""

    def test_shape(self):
        J = planar_2r_jacobian(1.0, 1.0, 0, 0)
        assert J.shape == (2, 2)

    def test_fully_extended(self):
        """At theta1=0, theta2=0, Jacobian has known values."""
        L1, L2 = 1.0, 1.0
        J = planar_2r_jacobian(L1, L2, 0, 0)
        # J = | 0  0 |
        #     | 2  1 |
        expected = np.array([[0, 0], [L1 + L2, L2]])
        np.testing.assert_array_almost_equal(J, expected)

    def test_matches_numerical(self):
        """Analytic Jacobian should match finite-difference approximation."""
        L1, L2 = 1.5, 1.0
        t1, t2 = 0.5, 0.8

        def fk_pos(q):
            res = planar_2r_fk(L1, L2, q[0], q[1])
            return res['end_effector']

        J_analytic = planar_2r_jacobian(L1, L2, t1, t2)
        J_numerical = numerical_jacobian(fk_pos, np.array([t1, t2]))
        np.testing.assert_array_almost_equal(J_analytic, J_numerical, decimal=5)

    def test_singular_at_full_extension(self):
        """At theta2 = 0, the arm is fully extended -> singular."""
        L1, L2 = 1.0, 1.0
        J = planar_2r_jacobian(L1, L2, 0, 0)
        assert np.abs(np.linalg.det(J)) < 1e-10

    def test_singular_at_fold(self):
        """At theta2 = pi, the arm is folded -> singular."""
        L1, L2 = 1.0, 1.0
        J = planar_2r_jacobian(L1, L2, 0, np.pi)
        assert np.abs(np.linalg.det(J)) < 1e-10

    def test_non_singular_away_from_boundary(self):
        """At theta2 = pi/2, arm should not be singular."""
        L1, L2 = 1.0, 1.0
        J = planar_2r_jacobian(L1, L2, 0, np.pi / 2)
        assert np.abs(np.linalg.det(J)) > 0.1


class TestPlanar3RJacobian:
    """Tests for 3R planar arm Jacobian."""

    def test_shape(self):
        J = planar_3r_jacobian(1.0, 1.0, 1.0, 0, 0, 0)
        assert J.shape == (3, 3)

    def test_orientation_row(self):
        """The orientation row should be all ones (each joint contributes equally)."""
        J = planar_3r_jacobian(1.0, 1.0, 1.0, 0.3, 0.4, 0.5)
        np.testing.assert_array_almost_equal(J[2, :], [1, 1, 1])

    def test_matches_numerical(self):
        """Analytic Jacobian should match numerical estimate."""
        L1, L2, L3 = 1.0, 1.5, 0.8
        t1, t2, t3 = 0.3, 0.7, -0.2

        def fk_pos_orient(q):
            res = planar_3r_fk(L1, L2, L3, q[0], q[1], q[2])
            return np.append(res['end_effector'], res['orientation'])

        J_analytic = planar_3r_jacobian(L1, L2, L3, t1, t2, t3)
        J_numerical = numerical_jacobian(fk_pos_orient, np.array([t1, t2, t3]))
        np.testing.assert_array_almost_equal(J_analytic, J_numerical, decimal=5)


class TestNumericalJacobian:
    """Tests for finite-difference Jacobian estimation."""

    def test_linear_function(self):
        """For f(q) = A @ q, the Jacobian should be A."""
        A = np.array([[1, 2], [3, 4], [5, 6]])
        J = numerical_jacobian(lambda q: A @ q, np.array([1.0, 2.0]))
        np.testing.assert_array_almost_equal(J, A, decimal=5)

    def test_shape(self):
        """Output shape should be (m, n)."""
        def f(q):
            return np.array([q[0]**2, q[1] * q[2]])
        J = numerical_jacobian(f, np.array([1.0, 2.0, 3.0]))
        assert J.shape == (2, 3)

    def test_quadratic_function(self):
        """For f(q) = [q0^2, q1^2], J = diag(2*q)."""
        q = np.array([3.0, 5.0])
        J = numerical_jacobian(lambda q: q**2, q)
        np.testing.assert_array_almost_equal(J, np.diag(2 * q), decimal=5)


class TestManipulability:
    """Tests for Yoshikawa's manipulability measure."""

    def test_identity_jacobian(self):
        """Manipulability of identity matrix should be 1."""
        w = manipulability(np.eye(2))
        np.testing.assert_almost_equal(w, 1.0)

    def test_singular_jacobian(self):
        """Manipulability should be 0 at singularity."""
        J = np.array([[1, 1], [1, 1]])  # Rank 1
        w = manipulability(J)
        np.testing.assert_almost_equal(w, 0.0)

    def test_non_negative(self):
        """Manipulability should always be >= 0."""
        np.random.seed(42)
        for _ in range(20):
            J = np.random.randn(2, 3)
            assert manipulability(J) >= -1e-10

    def test_scaled_jacobian(self):
        """Scaling J by c multiplies manipulability by |c|^m (m=rows)."""
        J = np.array([[1.0, 0], [0, 2.0]])
        w1 = manipulability(J)
        w2 = manipulability(3 * J)
        np.testing.assert_almost_equal(w2, 9 * w1)  # |3|^2 = 9 for 2x2


class TestIsSingular:
    """Tests for singularity detection."""

    def test_rank_deficient(self):
        """Rank-deficient matrix should be singular."""
        J = np.array([[1, 2], [2, 4]])  # Rank 1
        assert is_singular(J) is True

    def test_zero_matrix(self):
        """Zero matrix should be singular."""
        assert is_singular(np.zeros((2, 2))) is True

    def test_identity_not_singular(self):
        """Identity should not be singular."""
        assert is_singular(np.eye(3)) is False

    def test_well_conditioned(self):
        """A well-conditioned random matrix should not be singular."""
        np.random.seed(42)
        J = np.eye(3) + 0.1 * np.random.randn(3, 3)
        assert is_singular(J) is False


class TestConditionNumber:
    """Tests for Jacobian condition number."""

    def test_identity(self):
        """Condition number of identity is 1."""
        kappa = condition_number(np.eye(3))
        np.testing.assert_almost_equal(kappa, 1.0)

    def test_singular_is_inf(self):
        """Condition number of singular matrix should be inf."""
        J = np.array([[1, 1], [1, 1]])
        assert condition_number(J) == np.inf

    def test_diagonal(self):
        """Condition number of diag(a, b) = max/min."""
        J = np.diag([10.0, 2.0])
        np.testing.assert_almost_equal(condition_number(J), 5.0)


class TestPseudoinverseIK:
    """Tests for pseudo-inverse IK step."""

    def test_zero_error(self):
        """Zero error should give zero step."""
        J = np.eye(2)
        x = np.array([1, 2])
        dq = pseudoinverse_ik_step(J, x, x)
        np.testing.assert_array_almost_equal(dq, np.zeros(2))

    def test_identity_jacobian(self):
        """With identity Jacobian, delta_q = error."""
        J = np.eye(3)
        x_des = np.array([1, 2, 3])
        x_cur = np.array([0, 0, 0])
        dq = pseudoinverse_ik_step(J, x_des, x_cur)
        np.testing.assert_array_almost_equal(dq, x_des)

    def test_gain(self):
        """Gain < 1 should scale the step."""
        J = np.eye(2)
        x_des = np.array([4, 6])
        x_cur = np.array([0, 0])
        dq = pseudoinverse_ik_step(J, x_des, x_cur, gain=0.5)
        np.testing.assert_array_almost_equal(dq, [2, 3])

    def test_direction_correct(self):
        """Step should reduce error."""
        L1, L2 = 1.0, 1.0
        t1, t2 = 0.3, 0.8
        J = planar_2r_jacobian(L1, L2, t1, t2)
        x_cur = planar_2r_fk(L1, L2, t1, t2)['end_effector']
        x_des = x_cur + np.array([0.01, 0.01])
        dq = pseudoinverse_ik_step(J, x_des, x_cur, gain=1.0)
        # After step, J @ dq should approximately equal the error
        np.testing.assert_array_almost_equal(J @ dq, x_des - x_cur, decimal=5)


class TestDampedPseudoinverseIK:
    """Tests for damped least-squares IK."""

    def test_zero_error(self):
        """Zero error should give zero step."""
        J = np.eye(2)
        x = np.array([1, 2])
        dq = damped_pseudoinverse_ik_step(J, x, x)
        np.testing.assert_array_almost_equal(dq, np.zeros(2))

    def test_bounded_near_singularity(self):
        """Damped IK should not blow up near singularities."""
        # Near-singular Jacobian
        J = np.array([[1, 1], [1, 1.0001]])
        x_des = np.array([1, 0])
        x_cur = np.array([0, 0])
        dq = damped_pseudoinverse_ik_step(J, x_des, x_cur, damping=0.1)
        # Should be finite
        assert np.all(np.isfinite(dq))
        assert np.linalg.norm(dq) < 100

    def test_converges_to_pseudoinverse_with_zero_damping(self):
        """With damping -> 0, should approach pseudo-inverse result."""
        J = np.array([[1, 0], [0, 2]])
        x_des = np.array([3, 4])
        x_cur = np.array([0, 0])
        dq_damped = damped_pseudoinverse_ik_step(J, x_des, x_cur, damping=1e-10)
        dq_pinv = pseudoinverse_ik_step(J, x_des, x_cur)
        np.testing.assert_array_almost_equal(dq_damped, dq_pinv, decimal=5)


class TestNullSpaceProjection:
    """Tests for null-space projection."""

    def test_square_full_rank(self):
        """Full-rank square Jacobian has no null space -> projection is zero."""
        J = np.eye(3)
        q_null = np.array([1, 2, 3])
        projected = null_space_projection(J, q_null)
        np.testing.assert_array_almost_equal(projected, np.zeros(3))

    def test_redundant_system(self):
        """For a 2x3 Jacobian (redundant), null space should be 1D."""
        J = np.array([[1, 0, 0], [0, 1, 0]])  # 2x3, null space = [0, 0, z]
        q_null = np.array([1, 1, 1])
        projected = null_space_projection(J, q_null)
        # Should only have z-component
        np.testing.assert_almost_equal(projected[0], 0, decimal=5)
        np.testing.assert_almost_equal(projected[1], 0, decimal=5)
        np.testing.assert_almost_equal(projected[2], 1.0, decimal=5)

    def test_projection_is_in_null_space(self):
        """J @ projected should be zero (no task-space effect)."""
        np.random.seed(42)
        J = np.random.randn(2, 4)
        q_null = np.random.randn(4)
        projected = null_space_projection(J, q_null)
        np.testing.assert_array_almost_equal(J @ projected, np.zeros(2), decimal=5)

    def test_idempotent(self):
        """Projecting twice should give same result."""
        J = np.array([[1, 0, 1], [0, 1, 0]])
        q_null = np.array([1, 2, 3])
        p1 = null_space_projection(J, q_null)
        p2 = null_space_projection(J, p1)
        np.testing.assert_array_almost_equal(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
