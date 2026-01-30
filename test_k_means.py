import numpy as np
import pytest
from k_means import cluster


def test_basic_clustering():
    """Test that k-means correctly identifies well-separated clusters."""
    # Create 3 well-separated clusters
    np.random.seed(42)
    cluster1 = np.random.randn(10, 2) + np.array([0, 0])
    cluster2 = np.random.randn(10, 2) + np.array([10, 10])
    cluster3 = np.random.randn(10, 2) + np.array([20, 0])

    points = np.vstack([cluster1, cluster2, cluster3])

    centroids, assignments = cluster(3, points)

    # Check that we get 3 centroids
    assert centroids.shape == (3, 2)

    # Check that assignments has shape (N,) with cluster indices
    assert assignments.shape == (30,)

    # All assignments should be valid cluster indices
    assert np.all((assignments >= 0) & (assignments < 3))


def test_two_clusters():
    """Test k-means with 2 simple clusters."""
    points = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [10, 10],
        [11, 10],
        [10, 11]
    ], dtype=float)

    centroids, assignments = cluster(2, points)

    assert centroids.shape == (2, 2)
    assert assignments.shape == (6,)

    # Points 0-2 should be in one cluster, points 3-5 in another
    assert assignments[0] == assignments[1] == assignments[2]
    assert assignments[3] == assignments[4] == assignments[5]
    assert assignments[0] != assignments[3]


def test_centroid_locations():
    """Test that centroids are computed correctly."""
    # Two obvious clusters
    points = np.array([
        [0, 0],
        [2, 0],
        [1, 1],
        [100, 100],
        [102, 100],
        [101, 101]
    ], dtype=float)

    centroids, assignments = cluster(2, points)
    assert assignments.shape == (6,)

    # Expected centroids are approximately (1, 0.33) and (101, 100.33)
    expected1 = np.array([1, 1/3])
    expected2 = np.array([101, 100 + 1/3])

    # Check that both expected centroids are found (order may vary)
    centroids_sorted = centroids[np.argsort(centroids[:, 0])]
    np.testing.assert_array_almost_equal(centroids_sorted[0], expected1, decimal=5)
    np.testing.assert_array_almost_equal(centroids_sorted[1], expected2, decimal=5)


def test_single_cluster():
    """Test k-means with k=1."""
    points = np.array([
        [0, 0],
        [2, 2],
        [4, 4]
    ], dtype=float)

    centroids, assignments = cluster(1, points)

    assert centroids.shape == (1, 2)
    assert assignments.shape == (3,)
    assert np.all(assignments == 0)  # All points in cluster 0
    expected_centroid = np.array([2, 2])
    np.testing.assert_array_almost_equal(centroids[0], expected_centroid)


def test_high_dimensional():
    """Test k-means with higher dimensional data."""
    np.random.seed(123)
    cluster1 = np.random.randn(20, 5) + np.array([0, 0, 0, 0, 0])
    cluster2 = np.random.randn(20, 5) + np.array([10, 10, 10, 10, 10])

    points = np.vstack([cluster1, cluster2])

    centroids, assignments = cluster(2, points)

    assert centroids.shape == (2, 5)
    assert assignments.shape == (40,)
    assert np.all((assignments >= 0) & (assignments < 2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])