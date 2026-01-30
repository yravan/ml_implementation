import numpy as np
import pytest
from point_cloud.normals import estimate_normals


class TestEstimateNormals:
    """Tests for point cloud normal estimation."""

    def test_shape(self):
        points = np.random.randn(50, 3)
        normals = estimate_normals(points, k=10)
        assert normals.shape == (50, 3)

    def test_unit_length(self):
        """All normals should be unit vectors."""
        points = np.random.randn(30, 3)
        normals = estimate_normals(points, k=10)
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_array_almost_equal(lengths, np.ones(30), decimal=5)

    def test_planar_points(self):
        """For points on the xy-plane, normals should be ±z."""
        np.random.seed(42)
        N = 50
        xy = np.random.randn(N, 2)
        z = np.zeros((N, 1))
        points = np.hstack([xy, z])
        normals = estimate_normals(points, k=10)
        # Normal should be [0, 0, ±1] for each point
        for i in range(N):
            assert abs(abs(normals[i, 2]) - 1.0) < 0.1
            assert abs(normals[i, 0]) < 0.1
            assert abs(normals[i, 1]) < 0.1

    def test_tilted_plane(self):
        """For points on a tilted plane, normals should be perpendicular."""
        np.random.seed(42)
        N = 100
        # Plane: z = x + y -> normal ∝ [-1, -1, 1]
        xy = np.random.randn(N, 2) * 2
        z = (xy[:, 0] + xy[:, 1]).reshape(-1, 1)
        points = np.hstack([xy, z])
        normals = estimate_normals(points, k=20)
        expected_dir = np.array([-1, -1, 1]) / np.sqrt(3)
        # Each normal should be ± expected_dir
        for i in range(N):
            alignment = abs(np.dot(normals[i], expected_dir))
            assert alignment > 0.8  # Approximately correct


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
