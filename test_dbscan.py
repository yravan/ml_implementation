import numpy as np
import pytest
from dbscan import dbscan


def test_basic_clustering():
    """Test that DBSCAN correctly identifies well-separated dense clusters."""
    np.random.seed(42)
    # Two dense clusters
    cluster1 = np.random.randn(20, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(20, 2) * 0.5 + np.array([10, 10])

    points = np.vstack([cluster1, cluster2])
    labels = dbscan(points, eps=1.5, min_pts=3)

    # Check output shape
    assert labels.shape == (40,)

    # Should have exactly 2 clusters (labels 0 and 1)
    unique_labels = set(labels[labels >= 0])
    assert len(unique_labels) == 2

    # Points 0-19 should be in one cluster, points 20-39 in another
    assert len(set(labels[:20])) == 1 and labels[0] >= 0
    assert len(set(labels[20:])) == 1 and labels[20] >= 0
    assert labels[0] != labels[20]


def test_noise_detection():
    """Test that isolated points are labeled as noise (-1)."""
    points = np.array([
        [0, 0],
        [0.5, 0],
        [0, 0.5],
        [0.5, 0.5],
        [100, 100],  # Isolated noise point
    ], dtype=float)

    labels = dbscan(points, eps=1.0, min_pts=3)

    assert labels.shape == (5,)
    # First 4 points should form a cluster
    assert labels[0] == labels[1] == labels[2] == labels[3]
    assert labels[0] >= 0
    # Last point should be noise
    assert labels[4] == -1


def test_single_cluster():
    """Test DBSCAN with all points in one dense cluster."""
    np.random.seed(123)
    points = np.random.randn(30, 2) * 0.5

    labels = dbscan(points, eps=2.0, min_pts=3)

    assert labels.shape == (30,)
    # All points should be in the same cluster
    assert np.all(labels >= 0)
    assert len(set(labels)) == 1


def test_all_noise():
    """Test DBSCAN when all points are too sparse to form clusters."""
    points = np.array([
        [0, 0],
        [10, 10],
        [20, 20],
        [30, 30],
    ], dtype=float)

    labels = dbscan(points, eps=1.0, min_pts=3)

    assert labels.shape == (4,)
    # All points should be noise
    assert np.all(labels == -1)


def test_min_pts_threshold():
    """Test that min_pts correctly determines core points."""
    points = np.array([
        [0, 0],
        [0.5, 0],
        [0, 0.5],
        [10, 10],
        [10.5, 10],
    ], dtype=float)

    # With min_pts=3, first group forms cluster, second doesn't
    labels = dbscan(points, eps=1.0, min_pts=3)
    assert labels[0] == labels[1] == labels[2]
    assert labels[0] >= 0
    assert labels[3] == -1 and labels[4] == -1

    # With min_pts=2, both groups should form clusters
    labels = dbscan(points, eps=1.0, min_pts=2)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] >= 0 and labels[3] >= 0


def test_chain_cluster():
    """Test that DBSCAN can find chain-shaped clusters."""
    # Points forming a chain where each is connected to neighbors
    points = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
    ], dtype=float)

    labels = dbscan(points, eps=1.5, min_pts=2)

    assert labels.shape == (5,)
    # All points should be in the same cluster
    assert np.all(labels >= 0)
    assert len(set(labels)) == 1


def test_high_dimensional():
    """Test DBSCAN with higher dimensional data."""
    np.random.seed(456)
    cluster1 = np.random.randn(15, 5) * 0.5
    cluster2 = np.random.randn(15, 5) * 0.5 + 10

    points = np.vstack([cluster1, cluster2])
    labels = dbscan(points, eps=2.0, min_pts=3)

    assert labels.shape == (30,)
    # Should detect 2 clusters
    unique_labels = set(labels[labels >= 0])
    assert len(unique_labels) == 2


def test_border_points():
    """Test that border points are assigned to a cluster."""
    # Core points form a tight cluster, border point is within eps of core but not itself core
    points = np.array([
        [0, 0],      # core
        [0.3, 0],    # core
        [0, 0.3],    # core
        [0.3, 0.3],  # core
        [0.7, 0],    # border point - within eps=0.5 of point 1 (dist=0.4), but has < 3 neighbors
    ], dtype=float)

    labels = dbscan(points, eps=0.5, min_pts=3)

    # All points should be in the same cluster (border included)
    assert np.all(labels >= 0)
    assert len(set(labels)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])