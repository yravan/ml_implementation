import numpy as np


def dbscan(points, eps, min_pts):
    """
    DBSCAN clustering algorithm.

    Parameters:
        points: np.ndarray of shape (N, d) - N points in d dimensions
        eps: float - maximum distance between two points to be considered neighbors
        min_pts: int - minimum number of points to form a dense region (core point)

    Returns:
        labels: np.ndarray of shape (N,) - cluster labels for each point
                -1 indicates noise, 0, 1, 2, ... indicate cluster IDs
    """

    n = points.shape[0]
    labels = np.full(n, -1)  # Initialize all points as noise (-1)
    cur_label = 0
    indices = np.arange(n)

    # TODO: Implement DBSCAN
    distances = np.linalg.norm(points[:, None, :]  - points[None, :, :], axis = -1) # N x N
    neighbors = (distances < eps).sum(axis=-1) # N
    core_points = indices[neighbors >= min_pts]
    queue = set(core_points.tolist())
    while queue:
        center = queue.pop()
        if labels[center] != -1:
            continue
        labels[center] = cur_label
        new_queue = [center]
        while new_queue:
            cur_core_point = new_queue.pop()
            close_point_mask = np.logical_and(distances[cur_core_point] <= eps, labels == -1)
            new_queue.extend(indices[np.logical_and(neighbors >= min_pts, close_point_mask)].tolist())
            labels[close_point_mask] = cur_label
        cur_label += 1


    return labels

def generate_moons(n_points=200, noise=0.1, seed=42):
    """Generate two interleaving half circles (moons)."""
    np.random.seed(seed)
    n = n_points // 2
    theta1 = np.linspace(0, np.pi, n)
    x1, y1 = np.cos(theta1), np.sin(theta1)
    theta2 = np.linspace(0, np.pi, n)
    x2, y2 = 1 - np.cos(theta2), 0.5 - np.sin(theta2)
    points = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    return points + np.random.randn(*points.shape) * noise


def generate_blobs(n_points=150, n_clusters=3, std=0.5, seed=42):
    """Generate well-separated blob clusters."""
    np.random.seed(seed)
    centers = np.random.randn(n_clusters, 2) * 5
    points = [np.random.randn(n_points // n_clusters, 2) * std + c for c in centers]
    return np.vstack(points)


def main():
    import matplotlib.pyplot as plt

    # Generate datasets
    blobs = generate_blobs()
    moons = generate_moons()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Row 1: Original data
    axes[0, 0].scatter(blobs[:, 0], blobs[:, 1], s=30, alpha=0.7)
    axes[0, 0].set_title("Blobs (Original)")
    axes[0, 0].set_aspect('equal')

    axes[0, 1].scatter(moons[:, 0], moons[:, 1], s=30, alpha=0.7)
    axes[0, 1].set_title("Moons (Original)")
    axes[0, 1].set_aspect('equal')

    # Row 2: DBSCAN results
    for ax, points, eps, name in [
        (axes[1, 0], blobs, 1.0, "Blobs"),
        (axes[1, 1], moons, 0.2, "Moons"),
    ]:
        labels = dbscan(points, eps=eps, min_pts=5)
        noise_mask = labels == -1
        ax.scatter(points[~noise_mask, 0], points[~noise_mask, 1],
                   c=labels[~noise_mask], cmap='tab10', s=30, alpha=0.7)
        if np.any(noise_mask):
            ax.scatter(points[noise_mask, 0], points[noise_mask, 1],
                       c='grey', marker='x', s=30, label='noise')
            ax.legend()
        ax.set_title(f"DBSCAN on {name} (eps={eps}, min_pts=5)")
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig("dbscan_results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
