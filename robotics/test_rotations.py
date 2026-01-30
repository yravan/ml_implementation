import numpy as np
import pytest
from robotics.rotations import (
    rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
    rpy_to_rotation_matrix, rotation_matrix_to_rpy,
    axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle,
    quaternion_to_rotation_matrix, rotation_matrix_to_quaternion,
    quaternion_multiply, quaternion_conjugate,
    slerp, is_valid_rotation_matrix,
)


class TestElementaryRotations:
    """Tests for Rx, Ry, Rz rotation matrices."""

    def test_rx_zero(self):
        """Rx(0) should be identity."""
        R = rotation_matrix_x(0)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_ry_zero(self):
        """Ry(0) should be identity."""
        R = rotation_matrix_y(0)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_rz_zero(self):
        """Rz(0) should be identity."""
        R = rotation_matrix_z(0)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_rx_90(self):
        """Rx(pi/2) should map y->z, z->-y."""
        R = rotation_matrix_x(np.pi / 2)
        # x stays x
        np.testing.assert_array_almost_equal(R @ [1, 0, 0], [1, 0, 0])
        # y -> z
        np.testing.assert_array_almost_equal(R @ [0, 1, 0], [0, 0, 1])
        # z -> -y
        np.testing.assert_array_almost_equal(R @ [0, 0, 1], [0, -1, 0])

    def test_ry_90(self):
        """Ry(pi/2) should map z->x, x->-z."""
        R = rotation_matrix_y(np.pi / 2)
        np.testing.assert_array_almost_equal(R @ [0, 0, 1], [1, 0, 0])
        np.testing.assert_array_almost_equal(R @ [1, 0, 0], [0, 0, -1])

    def test_rz_90(self):
        """Rz(pi/2) should map x->y, y->-x."""
        R = rotation_matrix_z(np.pi / 2)
        np.testing.assert_array_almost_equal(R @ [1, 0, 0], [0, 1, 0])
        np.testing.assert_array_almost_equal(R @ [0, 1, 0], [-1, 0, 0])

    def test_all_valid_so3(self):
        """All elementary rotations should be valid SO(3) matrices."""
        for theta in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi]:
            for rot_func in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
                R = rot_func(theta)
                assert R.shape == (3, 3)
                np.testing.assert_array_almost_equal(R.T @ R, np.eye(3))
                assert np.abs(np.linalg.det(R) - 1.0) < 1e-6

    def test_rx_inverse(self):
        """Rx(θ) @ Rx(-θ) = I."""
        theta = np.pi / 5
        R = rotation_matrix_x(theta) @ rotation_matrix_x(-theta)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_rz_composition(self):
        """Rz(a) @ Rz(b) = Rz(a+b)."""
        a, b = np.pi / 3, np.pi / 6
        R1 = rotation_matrix_z(a) @ rotation_matrix_z(b)
        R2 = rotation_matrix_z(a + b)
        np.testing.assert_array_almost_equal(R1, R2)


class TestRPY:
    """Tests for Roll-Pitch-Yaw <-> rotation matrix conversions."""

    def test_identity(self):
        """RPY (0, 0, 0) should give identity."""
        R = rpy_to_rotation_matrix(0, 0, 0)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_pure_roll(self):
        """Pure roll should be Rx."""
        theta = np.pi / 4
        R = rpy_to_rotation_matrix(theta, 0, 0)
        R_expected = rotation_matrix_x(theta)
        np.testing.assert_array_almost_equal(R, R_expected)

    def test_pure_pitch(self):
        """Pure pitch should be Ry."""
        theta = np.pi / 6
        R = rpy_to_rotation_matrix(0, theta, 0)
        R_expected = rotation_matrix_y(theta)
        np.testing.assert_array_almost_equal(R, R_expected)

    def test_pure_yaw(self):
        """Pure yaw should be Rz."""
        theta = np.pi / 3
        R = rpy_to_rotation_matrix(0, 0, theta)
        R_expected = rotation_matrix_z(theta)
        np.testing.assert_array_almost_equal(R, R_expected)

    def test_roundtrip(self):
        """Converting RPY -> R -> RPY should recover original angles."""
        roll, pitch, yaw = 0.3, 0.2, 0.5
        R = rpy_to_rotation_matrix(roll, pitch, yaw)
        r2, p2, y2 = rotation_matrix_to_rpy(R)
        np.testing.assert_almost_equal(r2, roll, decimal=6)
        np.testing.assert_almost_equal(p2, pitch, decimal=6)
        np.testing.assert_almost_equal(y2, yaw, decimal=6)

    def test_roundtrip_negative_angles(self):
        """Roundtrip with negative angles."""
        roll, pitch, yaw = -0.4, 0.1, -0.6
        R = rpy_to_rotation_matrix(roll, pitch, yaw)
        r2, p2, y2 = rotation_matrix_to_rpy(R)
        # Verify the reconstructed rotation matches
        R2 = rpy_to_rotation_matrix(r2, p2, y2)
        np.testing.assert_array_almost_equal(R, R2)

    def test_composition_order(self):
        """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
        roll, pitch, yaw = 0.3, 0.4, 0.5
        R = rpy_to_rotation_matrix(roll, pitch, yaw)
        R_expected = rotation_matrix_z(yaw) @ rotation_matrix_y(pitch) @ rotation_matrix_x(roll)
        np.testing.assert_array_almost_equal(R, R_expected)


class TestAxisAngle:
    """Tests for axis-angle <-> rotation matrix conversions."""

    def test_zero_angle(self):
        """Zero angle should give identity regardless of axis."""
        R = axis_angle_to_rotation_matrix(np.array([1, 0, 0]), 0.0)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_x_axis_matches_rx(self):
        """Rotation about x-axis should match Rx."""
        theta = np.pi / 3
        R = axis_angle_to_rotation_matrix(np.array([1, 0, 0]), theta)
        np.testing.assert_array_almost_equal(R, rotation_matrix_x(theta))

    def test_y_axis_matches_ry(self):
        """Rotation about y-axis should match Ry."""
        theta = np.pi / 4
        R = axis_angle_to_rotation_matrix(np.array([0, 1, 0]), theta)
        np.testing.assert_array_almost_equal(R, rotation_matrix_y(theta))

    def test_z_axis_matches_rz(self):
        """Rotation about z-axis should match Rz."""
        theta = np.pi / 6
        R = axis_angle_to_rotation_matrix(np.array([0, 0, 1]), theta)
        np.testing.assert_array_almost_equal(R, rotation_matrix_z(theta))

    def test_normalizes_axis(self):
        """Non-unit axis should be normalized internally."""
        R1 = axis_angle_to_rotation_matrix(np.array([0, 0, 5]), np.pi / 4)
        R2 = axis_angle_to_rotation_matrix(np.array([0, 0, 1]), np.pi / 4)
        np.testing.assert_array_almost_equal(R1, R2)

    def test_roundtrip(self):
        """axis-angle -> R -> axis-angle should recover original."""
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        angle = np.pi / 3
        R = axis_angle_to_rotation_matrix(axis, angle)
        axis2, angle2 = rotation_matrix_to_axis_angle(R)
        np.testing.assert_almost_equal(angle2, angle, decimal=6)
        np.testing.assert_array_almost_equal(axis2, axis, decimal=5)

    def test_pi_rotation(self):
        """180-degree rotation should produce valid result."""
        axis = np.array([0, 0, 1])
        R = axis_angle_to_rotation_matrix(axis, np.pi)
        axis2, angle2 = rotation_matrix_to_axis_angle(R)
        np.testing.assert_almost_equal(angle2, np.pi, decimal=5)
        # For pi rotation, axis could be ±original
        assert np.allclose(np.abs(axis2), np.abs(axis), atol=1e-5)

    def test_produces_valid_so3(self):
        """Rodrigues' formula should always produce valid SO(3)."""
        np.random.seed(42)
        for _ in range(10):
            axis = np.random.randn(3)
            angle = np.random.uniform(0, np.pi)
            R = axis_angle_to_rotation_matrix(axis, angle)
            np.testing.assert_array_almost_equal(R.T @ R, np.eye(3))
            np.testing.assert_almost_equal(np.linalg.det(R), 1.0)


class TestQuaternion:
    """Tests for quaternion <-> rotation matrix conversions."""

    def test_identity_quaternion(self):
        """q = [1, 0, 0, 0] should give identity rotation."""
        R = quaternion_to_rotation_matrix(np.array([1, 0, 0, 0], dtype=float))
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_90_about_z(self):
        """Quaternion for 90-deg rotation about z: q = [cos(45), 0, 0, sin(45)]."""
        q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        R = quaternion_to_rotation_matrix(q)
        R_expected = rotation_matrix_z(np.pi / 2)
        np.testing.assert_array_almost_equal(R, R_expected)

    def test_roundtrip(self):
        """R -> q -> R should recover original rotation."""
        R_orig = rpy_to_rotation_matrix(0.3, 0.5, 0.7)
        q = rotation_matrix_to_quaternion(R_orig)
        R_back = quaternion_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R_back, R_orig)

    def test_quaternion_is_unit(self):
        """rotation_matrix_to_quaternion should return unit quaternion."""
        np.random.seed(42)
        for _ in range(10):
            axis = np.random.randn(3)
            angle = np.random.uniform(0, np.pi)
            R = axis_angle_to_rotation_matrix(axis, angle)
            q = rotation_matrix_to_quaternion(R)
            np.testing.assert_almost_equal(np.linalg.norm(q), 1.0)

    def test_w_non_negative(self):
        """Convention: returned quaternion should have w >= 0."""
        R = rotation_matrix_z(2.5)  # Large angle
        q = rotation_matrix_to_quaternion(R)
        assert q[0] >= -1e-10  # w >= 0

    def test_multiply_identity(self):
        """q * identity = q."""
        q = np.array([np.cos(0.3), np.sin(0.3), 0, 0])
        q_id = np.array([1.0, 0, 0, 0])
        result = quaternion_multiply(q, q_id)
        np.testing.assert_array_almost_equal(result, q)

    def test_multiply_matches_rotation_composition(self):
        """q1 * q2 should correspond to R1 @ R2."""
        R1 = rotation_matrix_x(0.5)
        R2 = rotation_matrix_z(0.7)
        q1 = rotation_matrix_to_quaternion(R1)
        q2 = rotation_matrix_to_quaternion(R2)
        q12 = quaternion_multiply(q1, q2)

        R_from_q = quaternion_to_rotation_matrix(q12)
        R_direct = R1 @ R2
        np.testing.assert_array_almost_equal(R_from_q, R_direct)

    def test_conjugate(self):
        """q * q_conjugate should give identity quaternion."""
        q = np.array([0.5, 0.5, 0.5, 0.5])  # Unit quaternion
        q_conj = quaternion_conjugate(q)
        result = quaternion_multiply(q, q_conj)
        np.testing.assert_array_almost_equal(result, [1, 0, 0, 0])

    def test_conjugate_sign(self):
        """Conjugate should negate vector part."""
        q = np.array([0.1, 0.2, 0.3, 0.4])
        q_conj = quaternion_conjugate(q)
        np.testing.assert_almost_equal(q_conj[0], q[0])
        np.testing.assert_array_almost_equal(q_conj[1:], -q[1:])


class TestSlerp:
    """Tests for spherical linear interpolation."""

    def test_t0_returns_start(self):
        """slerp(q0, q1, 0) = q0."""
        q0 = np.array([1, 0, 0, 0], dtype=float)
        q1 = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        result = slerp(q0, q1, 0.0)
        np.testing.assert_array_almost_equal(result, q0)

    def test_t1_returns_end(self):
        """slerp(q0, q1, 1) = q1 (up to sign)."""
        q0 = np.array([1, 0, 0, 0], dtype=float)
        q1 = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        result = slerp(q0, q1, 1.0)
        # q and -q represent the same rotation
        assert np.allclose(result, q1, atol=1e-6) or np.allclose(result, -q1, atol=1e-6)

    def test_midpoint_rotation(self):
        """Midpoint should be halfway rotation."""
        q0 = np.array([1, 0, 0, 0], dtype=float)  # Identity
        # 90 degrees about z
        q1 = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        mid = slerp(q0, q1, 0.5)
        R_mid = quaternion_to_rotation_matrix(mid)
        R_expected = rotation_matrix_z(np.pi / 4)  # 45 degrees
        np.testing.assert_array_almost_equal(R_mid, R_expected, decimal=5)

    def test_output_is_unit(self):
        """Slerp output should always be a unit quaternion."""
        q0 = np.array([1, 0, 0, 0], dtype=float)
        q1 = np.array([0.5, 0.5, 0.5, 0.5])
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = slerp(q0, q1, t)
            np.testing.assert_almost_equal(np.linalg.norm(result), 1.0)


class TestIsValidRotationMatrix:
    """Tests for SO(3) validation."""

    def test_identity_is_valid(self):
        """Identity is a valid rotation."""
        assert is_valid_rotation_matrix(np.eye(3)) is True

    def test_valid_rotation(self):
        """A proper rotation matrix should pass."""
        R = rotation_matrix_z(0.7)
        assert is_valid_rotation_matrix(R) is True

    def test_reflection_is_invalid(self):
        """A reflection (det = -1) is not a rotation."""
        R = np.diag([1, 1, -1])
        assert is_valid_rotation_matrix(R) is False

    def test_non_orthogonal_is_invalid(self):
        """A non-orthogonal matrix should fail."""
        M = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
        assert is_valid_rotation_matrix(M) is False

    def test_wrong_shape(self):
        """Non-3x3 matrix should fail."""
        assert is_valid_rotation_matrix(np.eye(4)) is False

    def test_scaled_rotation_is_invalid(self):
        """A scaled rotation (not orthonormal) should fail."""
        R = 2 * rotation_matrix_x(0.3)
        assert is_valid_rotation_matrix(R) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
