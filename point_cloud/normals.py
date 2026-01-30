"""
Point cloud normal estimation.

For each point, estimate the local surface normal by:
    1. Find k nearest neighbors.
    2. Compute the covariance matrix of the local neighborhood.
    3. The normal is the eigenvector corresponding to the smallest eigenvalue
       (the direction of least variance).
"""

import numpy as np


def estimate_normals(points, k=20):
    """
    Estimate surface normals for each point using local PCA.

    Algorithm for each point p_i:
        1. Find k nearest neighbors of p_i.
        2. Compute covariance: C = (1/k) Σ_j (p_j - p̄)(p_j - p̄)^T
        3. Eigendecompose C: the eigenvector with smallest eigenvalue
           is the normal direction.

    Note: Normal orientation (sign) is ambiguous and may need to be
    made consistent in a post-processing step.

    Parameters:
        points: np.ndarray of shape (N, 3) - Input point cloud.
        k: int - Number of neighbors for local PCA.

    Returns:
        normals: np.ndarray of shape (N, 3) - Estimated unit normals.
    """
    normals = None
    return normals
