"""
Dimensionality Reduction Module - Feature Space Transformation

Implementations:
- pca: Principal Component Analysis (linear, unsupervised)
- tsne: t-Distributed Stochastic Neighbor Embedding (nonlinear visualization)
- lda: Linear Discriminant Analysis (linear, supervised)

Each module includes comprehensive educational documentation with
theory, mathematics, algorithms, and complexity analysis.
"""

from . import pca
from . import tsne
from . import lda

__all__ = ["pca", "tsne", "lda"]
