"""
Clustering Module - Unsupervised Clustering Algorithms

Implementations:
- kmeans: K-Means and K-Means++ initialization
- gmm: Gaussian Mixture Model with EM algorithm
- dbscan: Density-Based Spatial Clustering
- spectral: Spectral Clustering

Each module includes comprehensive educational documentation.
"""

from . import kmeans
from . import gmm
from . import dbscan
from . import spectral

__all__ = ["kmeans", "gmm", "dbscan", "spectral"]
