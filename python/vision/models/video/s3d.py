"""
S3D - Separable 3D CNN
======================

From "Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification"
https://arxiv.org/abs/1712.04851

Key idea: Replace 3D convolutions with separable 3D convolutions.
    - Temporal conv: (t, 1, 1)
    - Spatial conv: (1, k, k)

Similar idea to (2+1)D but applied to Inception-style architecture.
"""

from python.nn_core import Module


class SepInceptionBlock3D(Module):
    """Separable 3D Inception block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # TODO: Implement separable inception block
        raise NotImplementedError("TODO: Implement SepInceptionBlock3D")


class S3D(Module):
    """
    S3D video classification model.

    Args:
        num_classes: Number of output classes
    """

    def __init__(self, num_classes: int = 400):
        super().__init__()
        # TODO: Implement S3D
        raise NotImplementedError("TODO: Implement S3D")

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("TODO: Implement forward")


def s3d(num_classes: int = 400, **kwargs) -> S3D:
    """S3D model."""
    return S3D(num_classes=num_classes, **kwargs)
