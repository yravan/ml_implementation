import numpy as np

def cluster(k, points):
    # points N x d
    cluster_indices = np.arange(k)
    new_clusters  = points[:k] # k x d
    old_cluster_assignment = np.zeros(points.shape[0])
    cluster_assignment = np.ones(points.shape[0])
    while np.array_equal(old_cluster_assignment - cluster_assignment):
        distances = np.linalg.norm(points[:, None, :] - clusters[None, ...], axis=-1) # N x k
        old_cluster_assignment = cluster_assignment
        cluster_assignment = np.argmin(distances, axis=-1) # N
        cluster_mask = cluster_assignment[None,...] == cluster_indices[:, None] # k x N
        empty = np.sum(cluster_mask, axis=-1) == 0
        cluster_mask[empty] = 1
        counts = np.sum(cluster_mask, axis=-1)
        new_clusters = (cluster_mask.astype(int) @ points) / counts[:, None]

    return new_clusters, cluster_assignment














