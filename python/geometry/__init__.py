"""
3D Geometry and Computer Vision Module.

This module implements fundamental concepts in 3D computer vision, including
camera models, geometric transformations, multi-view geometry, and differentiable rendering.

Submodules:
    - camera: Camera models and projections (pinhole, distortion)
    - transforms: 3D transformations (rotations, translations, SE(3))
    - multiview: Multi-view geometry (epipolar, triangulation, bundle adjustment)
    - correspondence: Feature matching and optical flow
    - rendering: Differentiable rendering (neural radiance, mesh rendering)

Key Concepts:
    - Homogeneous coordinates for unified transformation representation
    - Pinhole camera model: x = K[R|t]X
    - Epipolar geometry: x'^T F x = 0
    - Bundle adjustment: min Σ ||x_ij - π(K, R_i, t_i, X_j)||²

Implementation Status: STUB
Complexity: Advanced
Prerequisites: foundations, nn_core (for differentiable operations)

References:
    - "Multiple View Geometry in Computer Vision" (Hartley & Zisserman)
      https://www.robots.ox.ac.uk/~vgg/hzbook/
"""

from .camera import PinholeCamera, CameraIntrinsics, CameraDistortion
from .transforms import Rotation, Translation, RigidTransform, SE3
from .multiview import EpipolarGeometry, Triangulation, BundleAdjustment
from .correspondence import FeatureMatcher, OpticalFlow
from .rendering import DifferentiableRenderer, NeuralRadianceField

__all__ = [
    # Camera
    'PinholeCamera',
    'CameraIntrinsics',
    'CameraDistortion',
    # Transforms
    'Rotation',
    'Translation',
    'RigidTransform',
    'SE3',
    # Multi-view
    'EpipolarGeometry',
    'Triangulation',
    'BundleAdjustment',
    # Correspondence
    'FeatureMatcher',
    'OpticalFlow',
    # Rendering
    'DifferentiableRenderer',
    'NeuralRadianceField',
]
