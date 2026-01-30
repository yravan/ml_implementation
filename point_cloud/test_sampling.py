import numpy as np
import pytest
from point_cloud.sampling import farthest_point_sampling, knn_search, ball_query


class TestFarthestPointSampling:
    """Tests for FPS."""

    def test_correct_number(self):
        points = np.random.randn(100, 3)
        indices, sampled = farthest_point_sampling(points, 10)
        assert indices.shape == (10,)
        assert sampled.shape == (10, 3)

    def test_unique_indices(self):
        """All selected indices should be unique."""
        points = np.random.randn(50, 3)
        indices, _ = farthest_point_sampling(points, 20)
        assert len(set(indices)) == 20

    def test_valid_indices(self):
        """Indices should be valid into original array."""
        points = np.random.randn(30, 3)
        indices, sampled = farthest_point_sampling(points, 10)
        np.testing.assert_array_almost_equal(sampled, points[indices])

    def test_all_points(self):
        """Sampling N from N points should return all."""
        points = np.random.randn(5, 2)
        indices, _ = farthest_point_sampling(points, 5)
        assert set(indices) == {0, 1, 2, 3, 4}

    def test_well_distributed(self):
        """FPS should spread points out better than random."""
        np.random.seed(42)
        # Points on a line from 0 to 100
        points = np.linspace(0, 100, 101).reshape(-1, 1)
        indices, sampled = farthest_point_sampling(points, 11)
        # Should pick roughly 0, 10, 20, ..., 100
        diffs = np.sort(np.diff(np.sort(sampled.flatten())))
        # Minimum gap should be large (well-distributed)
        assert diffs[0] > 5  # Not clustered


class TestKNNSearch:
    """Tests for k-nearest neighbor search."""

    def test_shapes(self):
        query = np.random.randn(5, 3)
        reference = np.random.randn(20, 3)
        indices, distances = knn_search(query, reference, k=3)
        assert indices.shape == (5, 3)
        assert distances.shape == (5, 3)

    def test_self_query(self):
        """Querying a point against itself should give distance 0."""
        points = np.random.randn(10, 3)
        indices, distances = knn_search(points, points, k=1)
        np.testing.assert_array_almost_equal(distances[:, 0], np.zeros(10))

    def test_sorted_by_distance(self):
        """Neighbors should be sorted by distance."""
        np.random.seed(42)
        query = np.random.randn(3, 2)
        reference = np.random.randn(20, 2)
        _, distances = knn_search(query, reference, k=5)
        for i in range(3):
            assert np.all(np.diff(distances[i]) >= -1e-10)

    def test_correct_nearest(self):
        """Known nearest neighbor test."""
        reference = np.array([[0, 0], [1, 0], [0, 1], [10, 10]], dtype=float)
        query = np.array([[0.1, 0.1]], dtype=float)
        indices, _ = knn_search(query, reference, k=1)
        assert indices[0, 0] == 0  # Closest to origin

    def test_k_equals_n(self):
        """k = N should return all points."""
        query = np.array([[0, 0]], dtype=float)
        reference = np.random.randn(5, 2)
        indices, _ = knn_search(query, reference, k=5)
        assert set(indices[0]) == {0, 1, 2, 3, 4}


class TestBallQuery:
    """Tests for ball query."""

    def test_finds_close_points(self):
        reference = np.array([[0, 0], [0.5, 0], [10, 10]], dtype=float)
        query = np.array([[0, 0]], dtype=float)
        neighbors, distances = ball_query(query, reference, radius=1.0)
        # Should find points 0 and 1, not point 2
        assert 0 in neighbors[0]
        assert 1 in neighbors[0]
        assert 2 not in neighbors[0]

    def test_empty_ball(self):
        """No points within small radius."""
        reference = np.array([[10, 10]], dtype=float)
        query = np.array([[0, 0]], dtype=float)
        neighbors, _ = ball_query(query, reference, radius=1.0)
        assert len(neighbors[0]) == 0

    def test_all_in_ball(self):
        """Large radius should include all points."""
        reference = np.random.randn(10, 2) * 0.01
        query = np.array([[0, 0]], dtype=float)
        neighbors, _ = ball_query(query, reference, radius=100.0)
        assert len(neighbors[0]) == 10

    def test_max_k_limit(self):
        """max_k should cap the number of neighbors."""
        reference = np.random.randn(20, 2) * 0.1
        query = np.array([[0, 0]], dtype=float)
        neighbors, _ = ball_query(query, reference, radius=100.0, max_k=5)
        assert len(neighbors[0]) <= 5

    def test_distances_within_radius(self):
        """All returned distances should be <= radius."""
        np.random.seed(42)
        reference = np.random.randn(50, 3)
        query = np.random.randn(5, 3)
        radius = 1.5
        _, distances = ball_query(query, reference, radius=radius)
        for dists in distances:
            assert np.all(dists <= radius + 1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
