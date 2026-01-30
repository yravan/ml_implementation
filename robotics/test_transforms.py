import numpy as np
import pytest
from robotics.rotations import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z
from robotics.transforms import (
    make_transform, extract_rotation, extract_translation,
    transform_inverse, transform_compose, transform_point,
    transform_vector, skew_symmetric, adjoint_matrix,
)


class TestMakeTransform:
    """Tests for SE(3) construction."""

    def test_shape(self):
        """Output should be 4x4."""
        T = make_transform(np.eye(3), np.zeros(3))
        assert T.shape == (4, 4)

    def test_identity(self):
        """Identity rotation + zero translation = 4x4 identity."""
        T = make_transform(np.eye(3), np.zeros(3))
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_bottom_row(self):
        """Bottom row should always be [0, 0, 0, 1]."""
        R = rotation_matrix_z(0.5)
        p = np.array([1, 2, 3])
        T = make_transform(R, p)
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])

    def test_rotation_block(self):
        """Upper-left 3x3 should be R."""
        R = rotation_matrix_x(0.7)
        p = np.array([4, 5, 6])
        T = make_transform(R, p)
        np.testing.assert_array_almost_equal(T[:3, :3], R)

    def test_translation_column(self):
        """Right column (top 3) should be p."""
        R = np.eye(3)
        p = np.array([10, 20, 30])
        T = make_transform(R, p)
        np.testing.assert_array_almost_equal(T[:3, 3], p)


class TestExtract:
    """Tests for extracting R and p from T."""

    def test_extract_rotation(self):
        R = rotation_matrix_y(1.2)
        T = make_transform(R, np.array([1, 2, 3]))
        np.testing.assert_array_almost_equal(extract_rotation(T), R)

    def test_extract_translation(self):
        p = np.array([7, 8, 9])
        T = make_transform(np.eye(3), p)
        np.testing.assert_array_almost_equal(extract_translation(T), p)

    def test_extract_rotation_shape(self):
        T = make_transform(np.eye(3), np.zeros(3))
        assert extract_rotation(T).shape == (3, 3)

    def test_extract_translation_shape(self):
        T = make_transform(np.eye(3), np.zeros(3))
        assert extract_translation(T).shape == (3,)


class TestTransformInverse:
    """Tests for SE(3) inversion."""

    def test_identity_inverse(self):
        """Inverse of identity is identity."""
        T = np.eye(4)
        np.testing.assert_array_almost_equal(transform_inverse(T), np.eye(4))

    def test_inverse_composition_is_identity(self):
        """T @ T^{-1} = I."""
        R = rotation_matrix_z(0.8)
        p = np.array([1, 2, 3])
        T = make_transform(R, p)
        T_inv = transform_inverse(T)
        np.testing.assert_array_almost_equal(T @ T_inv, np.eye(4), decimal=10)

    def test_inverse_of_inverse(self):
        """(T^{-1})^{-1} = T."""
        R = rotation_matrix_x(1.1)
        p = np.array([-2, 5, 0.3])
        T = make_transform(R, p)
        np.testing.assert_array_almost_equal(
            transform_inverse(transform_inverse(T)), T
        )

    def test_pure_translation_inverse(self):
        """Inverse of pure translation negates the translation."""
        p = np.array([3, 4, 5])
        T = make_transform(np.eye(3), p)
        T_inv = transform_inverse(T)
        np.testing.assert_array_almost_equal(extract_translation(T_inv), -p)

    def test_matches_numpy_inv(self):
        """Should match np.linalg.inv for SE(3) matrices."""
        R = rotation_matrix_y(0.6) @ rotation_matrix_x(0.3)
        p = np.array([1, -1, 2])
        T = make_transform(R, p)
        np.testing.assert_array_almost_equal(
            transform_inverse(T), np.linalg.inv(T), decimal=10
        )


class TestTransformCompose:
    """Tests for SE(3) composition."""

    def test_single_transform(self):
        """Composing a single transform returns itself."""
        T = make_transform(rotation_matrix_z(0.5), np.array([1, 2, 3]))
        np.testing.assert_array_almost_equal(transform_compose(T), T)

    def test_two_translations(self):
        """Composing two pure translations adds them."""
        T1 = make_transform(np.eye(3), np.array([1, 0, 0]))
        T2 = make_transform(np.eye(3), np.array([0, 2, 0]))
        T = transform_compose(T1, T2)
        np.testing.assert_array_almost_equal(extract_translation(T), [1, 2, 0])

    def test_two_rotations(self):
        """Composing two pure rotations multiplies them."""
        R1 = rotation_matrix_z(0.3)
        R2 = rotation_matrix_z(0.4)
        T1 = make_transform(R1, np.zeros(3))
        T2 = make_transform(R2, np.zeros(3))
        T = transform_compose(T1, T2)
        R_expected = rotation_matrix_z(0.7)
        np.testing.assert_array_almost_equal(extract_rotation(T), R_expected)

    def test_compose_with_inverse_is_identity(self):
        """T @ T^{-1} via compose = identity."""
        T = make_transform(rotation_matrix_x(0.5), np.array([3, 4, 5]))
        T_inv = transform_inverse(T)
        result = transform_compose(T, T_inv)
        np.testing.assert_array_almost_equal(result, np.eye(4))

    def test_three_transforms(self):
        """Composing three transforms matches sequential multiplication."""
        T1 = make_transform(rotation_matrix_x(0.1), np.array([1, 0, 0]))
        T2 = make_transform(rotation_matrix_y(0.2), np.array([0, 1, 0]))
        T3 = make_transform(rotation_matrix_z(0.3), np.array([0, 0, 1]))
        result = transform_compose(T1, T2, T3)
        expected = T1 @ T2 @ T3
        np.testing.assert_array_almost_equal(result, expected)


class TestTransformPoint:
    """Tests for applying transforms to points."""

    def test_identity_transform(self):
        """Identity should not move the point."""
        T = np.eye(4)
        p = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(transform_point(T, p), p)

    def test_pure_translation(self):
        """Pure translation shifts the point."""
        T = make_transform(np.eye(3), np.array([10, 20, 30]))
        p = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(transform_point(T, p), [11, 22, 33])

    def test_pure_rotation(self):
        """Rotation about z by pi/2: (1,0,0) -> (0,1,0)."""
        T = make_transform(rotation_matrix_z(np.pi / 2), np.zeros(3))
        p = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(transform_point(T, p), [0, 1, 0])

    def test_batch_points(self):
        """Should handle (N, 3) batch of points."""
        T = make_transform(np.eye(3), np.array([1, 1, 1]))
        points = np.array([[0, 0, 0], [1, 2, 3], [-1, -2, -3]])
        result = transform_point(T, points)
        expected = points + np.array([1, 1, 1])
        np.testing.assert_array_almost_equal(result, expected)


class TestTransformVector:
    """Tests for applying transforms to direction vectors."""

    def test_translation_does_not_affect_vector(self):
        """Vectors should be unaffected by translation."""
        T = make_transform(np.eye(3), np.array([100, 200, 300]))
        v = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(transform_vector(T, v), v)

    def test_rotation_affects_vector(self):
        """Vectors should be rotated."""
        T = make_transform(rotation_matrix_z(np.pi / 2), np.array([10, 20, 30]))
        v = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(transform_vector(T, v), [0, 1, 0])

    def test_batch_vectors(self):
        """Should handle (N, 3) batch of vectors."""
        R = rotation_matrix_z(np.pi / 2)
        T = make_transform(R, np.array([99, 99, 99]))
        vecs = np.array([[1, 0, 0], [0, 1, 0]])
        result = transform_vector(T, vecs)
        expected = np.array([[0, 1, 0], [-1, 0, 0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestSkewSymmetric:
    """Tests for skew-symmetric matrix."""

    def test_shape(self):
        S = skew_symmetric(np.array([1, 2, 3]))
        assert S.shape == (3, 3)

    def test_antisymmetric(self):
        """S should equal -S^T."""
        v = np.array([1, 2, 3])
        S = skew_symmetric(v)
        np.testing.assert_array_almost_equal(S, -S.T)

    def test_cross_product(self):
        """[v]_x @ u should equal v x u."""
        v = np.array([1, 2, 3])
        u = np.array([4, 5, 6])
        S = skew_symmetric(v)
        np.testing.assert_array_almost_equal(S @ u, np.cross(v, u))

    def test_zero_vector(self):
        """Skew of zero vector is zero matrix."""
        S = skew_symmetric(np.zeros(3))
        np.testing.assert_array_almost_equal(S, np.zeros((3, 3)))


class TestAdjointMatrix:
    """Tests for the 6x6 adjoint representation."""

    def test_shape(self):
        T = make_transform(np.eye(3), np.zeros(3))
        Ad = adjoint_matrix(T)
        assert Ad.shape == (6, 6)

    def test_identity_is_identity(self):
        """Adjoint of identity should be 6x6 identity."""
        T = np.eye(4)
        Ad = adjoint_matrix(T)
        np.testing.assert_array_almost_equal(Ad, np.eye(6))

    def test_pure_rotation(self):
        """For pure rotation, Ad = block_diag(R, R)."""
        R = rotation_matrix_z(0.5)
        T = make_transform(R, np.zeros(3))
        Ad = adjoint_matrix(T)
        expected = np.zeros((6, 6))
        expected[:3, :3] = R
        expected[3:, 3:] = R
        np.testing.assert_array_almost_equal(Ad, expected)

    def test_composition_property(self):
        """Ad(T1 @ T2) = Ad(T1) @ Ad(T2)."""
        T1 = make_transform(rotation_matrix_x(0.3), np.array([1, 0, 0]))
        T2 = make_transform(rotation_matrix_y(0.5), np.array([0, 2, 0]))
        Ad_composed = adjoint_matrix(T1 @ T2)
        Ad_product = adjoint_matrix(T1) @ adjoint_matrix(T2)
        np.testing.assert_array_almost_equal(Ad_composed, Ad_product, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
