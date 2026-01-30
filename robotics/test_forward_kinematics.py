import numpy as np
import pytest
from robotics.forward_kinematics import (
    dh_transform, forward_kinematics_dh, planar_2r_fk, planar_3r_fk,
)


class TestDHTransform:
    """Tests for single-link DH transforms."""

    def test_identity_params(self):
        """All-zero DH params should give identity."""
        T = dh_transform(0, 0, 0, 0)
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_shape(self):
        T = dh_transform(0.5, 0.1, 0.3, 0.2)
        assert T.shape == (4, 4)

    def test_is_valid_se3(self):
        """DH transform should be a valid SE(3) matrix."""
        T = dh_transform(1.0, 0.5, 0.3, 0.8)
        # Bottom row
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])
        # Rotation part is SO(3)
        R = T[:3, :3]
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(3))
        np.testing.assert_almost_equal(np.linalg.det(R), 1.0)

    def test_pure_rotation_about_z(self):
        """theta-only DH: rotation about z-axis."""
        theta = np.pi / 4
        T = dh_transform(theta, 0, 0, 0)
        c, s = np.cos(theta), np.sin(theta)
        expected = np.eye(4)
        expected[:2, :2] = [[c, -s], [s, c]]
        np.testing.assert_array_almost_equal(T, expected)

    def test_pure_translation_along_z(self):
        """d-only DH: translation along z-axis."""
        d = 1.5
        T = dh_transform(0, d, 0, 0)
        np.testing.assert_array_almost_equal(T[:3, 3], [0, 0, d])
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))

    def test_pure_translation_along_x(self):
        """a-only DH: translation along x-axis."""
        a = 2.0
        T = dh_transform(0, 0, a, 0)
        np.testing.assert_array_almost_equal(T[:3, 3], [a, 0, 0])

    def test_known_values(self):
        """Verify against the closed-form DH matrix."""
        theta, d, a, alpha = np.pi / 6, 0.5, 1.0, np.pi / 3
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        expected = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d     ],
            [0,   0,        0,       1     ],
        ])
        T = dh_transform(theta, d, a, alpha)
        np.testing.assert_array_almost_equal(T, expected)


class TestForwardKinematicsDH:
    """Tests for general DH-based forward kinematics."""

    def test_single_revolute_joint(self):
        """Single revolute joint with offset angle 0."""
        # DH: [theta_offset=0, d=0, a=1, alpha=0]
        dh = np.array([[0, 0, 1.0, 0]])
        angles = np.array([np.pi / 2])
        T = forward_kinematics_dh(dh, angles)
        # Should translate 1 unit in the direction of theta=pi/2 -> (0, 1, 0)
        np.testing.assert_array_almost_equal(T[:3, 3], [0, 1, 0])

    def test_two_joint_planar(self):
        """Two revolute joints forming a 2R planar arm."""
        L1, L2 = 1.0, 1.0
        dh = np.array([
            [0, 0, L1, 0],
            [0, 0, L2, 0],
        ])
        # Both at 0 -> fully extended along x
        T = forward_kinematics_dh(dh, np.array([0, 0]))
        np.testing.assert_array_almost_equal(T[:3, 3], [L1 + L2, 0, 0])

    def test_two_joint_folded(self):
        """2R arm with theta2 = pi (folded back)."""
        L1, L2 = 1.0, 1.0
        dh = np.array([
            [0, 0, L1, 0],
            [0, 0, L2, 0],
        ])
        T = forward_kinematics_dh(dh, np.array([0, np.pi]))
        # Folded: end at (L1 - L2, 0, 0) = (0, 0, 0)
        np.testing.assert_array_almost_equal(T[:3, 3], [0, 0, 0], decimal=10)

    def test_zero_angles_give_identity_rotation_chain(self):
        """With alpha=0 and theta=0, rotations compose to identity."""
        dh = np.array([
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
        ])
        T = forward_kinematics_dh(dh, np.zeros(3))
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))
        np.testing.assert_array_almost_equal(T[:3, 3], [3, 0, 0])


class TestPlanar2RFK:
    """Tests for 2-link planar arm forward kinematics."""

    def test_both_zero(self):
        """Both angles zero: arm fully extended along x."""
        result = planar_2r_fk(1.0, 1.0, 0, 0)
        np.testing.assert_array_almost_equal(result['joint1'], [0, 0])
        np.testing.assert_array_almost_equal(result['joint2'], [1, 0])
        np.testing.assert_array_almost_equal(result['end_effector'], [2, 0])

    def test_theta1_90(self):
        """theta1 = pi/2, theta2 = 0: arm points up."""
        result = planar_2r_fk(1.0, 1.0, np.pi / 2, 0)
        np.testing.assert_array_almost_equal(result['joint2'], [0, 1])
        np.testing.assert_array_almost_equal(result['end_effector'], [0, 2])

    def test_theta2_90(self):
        """theta1 = 0, theta2 = pi/2: elbow bent up."""
        result = planar_2r_fk(1.0, 1.0, 0, np.pi / 2)
        np.testing.assert_array_almost_equal(result['joint2'], [1, 0])
        np.testing.assert_array_almost_equal(result['end_effector'], [1, 1])

    def test_folded_back(self):
        """theta2 = pi: arm folded back to origin."""
        result = planar_2r_fk(1.0, 1.0, 0, np.pi)
        np.testing.assert_array_almost_equal(result['end_effector'], [0, 0], decimal=10)

    def test_different_link_lengths(self):
        """Unequal links fully extended."""
        result = planar_2r_fk(2.0, 3.0, 0, 0)
        np.testing.assert_array_almost_equal(result['end_effector'], [5, 0])

    def test_workspace_radius(self):
        """End-effector distance should not exceed L1 + L2."""
        np.random.seed(42)
        L1, L2 = 1.5, 2.0
        for _ in range(20):
            t1 = np.random.uniform(-np.pi, np.pi)
            t2 = np.random.uniform(-np.pi, np.pi)
            result = planar_2r_fk(L1, L2, t1, t2)
            dist = np.linalg.norm(result['end_effector'])
            assert dist <= L1 + L2 + 1e-10

    def test_returns_transform(self):
        """T_02 should be a valid 4x4 matrix."""
        result = planar_2r_fk(1.0, 1.0, 0.3, 0.5)
        T = result['T_02']
        assert T.shape == (4, 4)
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])


class TestPlanar3RFK:
    """Tests for 3-link planar arm forward kinematics."""

    def test_all_zero(self):
        """All angles zero: arm fully extended along x."""
        result = planar_3r_fk(1.0, 1.0, 1.0, 0, 0, 0)
        np.testing.assert_array_almost_equal(result['end_effector'], [3, 0])
        np.testing.assert_almost_equal(result['orientation'], 0)

    def test_orientation_sums(self):
        """End-effector orientation should be theta1 + theta2 + theta3."""
        t1, t2, t3 = 0.3, 0.4, 0.5
        result = planar_3r_fk(1.0, 1.0, 1.0, t1, t2, t3)
        np.testing.assert_almost_equal(result['orientation'], t1 + t2 + t3)

    def test_joint_positions(self):
        """Joint positions should be correctly chained."""
        result = planar_3r_fk(1.0, 1.0, 1.0, np.pi / 2, 0, 0)
        np.testing.assert_array_almost_equal(result['joint1'], [0, 0])
        np.testing.assert_array_almost_equal(result['joint2'], [0, 1])
        np.testing.assert_array_almost_equal(result['joint3'], [0, 2])
        np.testing.assert_array_almost_equal(result['end_effector'], [0, 3])

    def test_folded_to_origin(self):
        """theta2 = 2pi/3, theta3 = 2pi/3 with equal links: forms equilateral return."""
        L = 1.0
        result = planar_3r_fk(L, L, L, 0, 2 * np.pi / 3, 2 * np.pi / 3)
        # Total angle = 4pi/3, forms a closed triangle
        # Verify end-effector is back near origin
        ee = result['end_effector']
        np.testing.assert_array_almost_equal(ee, [0, 0], decimal=10)

    def test_workspace_radius(self):
        """Distance should not exceed L1 + L2 + L3."""
        np.random.seed(42)
        L1, L2, L3 = 1.0, 1.5, 0.8
        for _ in range(20):
            t1 = np.random.uniform(-np.pi, np.pi)
            t2 = np.random.uniform(-np.pi, np.pi)
            t3 = np.random.uniform(-np.pi, np.pi)
            result = planar_3r_fk(L1, L2, L3, t1, t2, t3)
            dist = np.linalg.norm(result['end_effector'])
            assert dist <= L1 + L2 + L3 + 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
