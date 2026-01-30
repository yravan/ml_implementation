import numpy as np
import pytest
from point_cloud.registration import (
    find_correspondences, rigid_align_svd, compute_rmse, icp,
)


def apply_transform(points, R, t):
    """Apply rigid transform to points."""
    return (R @ points.T).T + t


class TestFindCorrespondences:
    """Tests for nearest-neighbor correspondences."""

    def test_shape(self):
        source = np.random.randn(10, 3)
        target = np.random.randn(20, 3)
        indices, distances = find_correspondences(source, target)
        assert indices.shape == (10,)
        assert distances.shape == (10,)

    def test_self_correspondences(self):
        """Points matched to themselves should have zero distance."""
        points = np.random.randn(10, 3)
        indices, distances = find_correspondences(points, points)
        np.testing.assert_array_equal(indices, np.arange(10))
        np.testing.assert_array_almost_equal(distances, np.zeros(10))

    def test_correct_nearest(self):
        source = np.array([[0, 0, 0]], dtype=float)
        target = np.array([[10, 0, 0], [1, 0, 0], [5, 0, 0]], dtype=float)
        indices, _ = find_correspondences(source, target)
        assert indices[0] == 1  # Closest

    def test_non_negative_distances(self):
        source = np.random.randn(5, 3)
        target = np.random.randn(10, 3)
        _, distances = find_correspondences(source, target)
        assert np.all(distances >= 0)


class TestRigidAlignSVD:
    """Tests for SVD-based rigid alignment (Procrustes)."""

    def test_identity_for_same_points(self):
        """Aligning a point cloud to itself: R=I, t=0."""
        points = np.random.randn(20, 3)
        R, t = rigid_align_svd(points, points)
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=5)
        np.testing.assert_array_almost_equal(t, np.zeros(3), decimal=5)

    def test_recovers_translation(self):
        """Should recover a known pure translation."""
        source = np.random.randn(30, 3)
        t_true = np.array([5, -3, 2])
        target = source + t_true
        R, t = rigid_align_svd(source, target)
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=5)
        np.testing.assert_array_almost_equal(t, t_true, decimal=5)

    def test_recovers_rotation(self):
        """Should recover a known pure rotation."""
        np.random.seed(42)
        source = np.random.randn(50, 3)
        theta = np.pi / 4
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1],
        ])
        target = (R_true @ source.T).T
        R, t = rigid_align_svd(source, target)
        np.testing.assert_array_almost_equal(R, R_true, decimal=5)
        np.testing.assert_array_almost_equal(t, np.zeros(3), decimal=5)

    def test_recovers_rigid_transform(self):
        """Should recover combined rotation + translation."""
        np.random.seed(42)
        source = np.random.randn(40, 3)
        theta = 0.3
        R_true = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)],
        ])
        t_true = np.array([1, 2, 3])
        target = apply_transform(source, R_true, t_true)
        R, t = rigid_align_svd(source, target)
        np.testing.assert_array_almost_equal(R, R_true, decimal=5)
        np.testing.assert_array_almost_equal(t, t_true, decimal=5)

    def test_rotation_is_so3(self):
        """Returned R should be a valid rotation (det=1, orthogonal)."""
        source = np.random.randn(20, 3)
        target = np.random.randn(20, 3)
        R, _ = rigid_align_svd(source, target)
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(3), decimal=5)
        np.testing.assert_almost_equal(np.linalg.det(R), 1.0, decimal=5)


class TestComputeRMSE:
    """Tests for RMSE computation."""

    def test_zero_for_perfect(self):
        source = np.random.randn(10, 3)
        rmse = compute_rmse(source, source, np.eye(3), np.zeros(3))
        np.testing.assert_almost_equal(rmse, 0.0)

    def test_known_rmse(self):
        source = np.array([[0, 0, 0]], dtype=float)
        target = np.array([[3, 4, 0]], dtype=float)
        rmse = compute_rmse(source, target, np.eye(3), np.zeros(3))
        np.testing.assert_almost_equal(rmse, 5.0)

    def test_non_negative(self):
        source = np.random.randn(10, 3)
        target = np.random.randn(10, 3)
        rmse = compute_rmse(source, target, np.eye(3), np.zeros(3))
        assert rmse >= 0


class TestICP:
    """Tests for Iterative Closest Point."""

    def test_known_translation(self):
        """ICP should recover a known translation."""
        np.random.seed(42)
        target = np.random.randn(50, 3)
        t_true = np.array([2, 0, 0])
        source = target + t_true  # source is target shifted
        R, t, transformed, n_iter = icp(source, target)
        # After ICP, transformed should be close to target
        rmse = compute_rmse(transformed, target, np.eye(3), np.zeros(3))
        assert rmse < 0.5

    def test_known_rotation(self):
        """ICP should recover a known rotation (small angle)."""
        np.random.seed(42)
        target = np.random.randn(100, 3)
        theta = 0.1  # Small angle
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1],
        ])
        source = (R_true @ target.T).T
        R, t, transformed, n_iter = icp(source, target)
        rmse = compute_rmse(transformed, target, np.eye(3), np.zeros(3))
        assert rmse < 0.5

    def test_converges(self):
        """ICP should converge in finite iterations."""
        np.random.seed(42)
        target = np.random.randn(30, 3)
        source = target + np.array([0.5, 0, 0])
        _, _, _, n_iter = icp(source, target, max_iter=100)
        assert n_iter > 0
        assert n_iter <= 100

    def test_output_shapes(self):
        source = np.random.randn(20, 3)
        target = np.random.randn(30, 3)
        R, t, transformed, n_iter = icp(source, target)
        assert R.shape == (3, 3)
        assert t.shape == (3,)
        assert transformed.shape == (20, 3)

    def test_identity_when_aligned(self):
        """Already-aligned clouds should give R≈I, t≈0."""
        points = np.random.randn(30, 3)
        R, t, _, _ = icp(points, points)
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=3)
        np.testing.assert_array_almost_equal(t, np.zeros(3), decimal=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
