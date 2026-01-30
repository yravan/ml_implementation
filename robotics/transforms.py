"""
Homogeneous transformations and SE(3) spatial algebra.

Covers rigid body transforms, composition, inversion, and the adjoint
representation used in spatial velocity transformations.

Reference: https://manipulation.csail.mit.edu/pick.html

Monogram notation:
    ^BX^A represents the pose of frame A measured from frame B.
    Composition: ^AX^C = ^AX^B @ ^BX^C  (intermediate frame B cancels)
    Inverse: ^BX^A = (^AX^B)^{-1}

Key concepts:
    - SE(3): The group of 4x4 homogeneous transforms  T = | R  p |
                                                           | 0  1 |
    - Points transform as: p' = R @ p + t
    - Vectors (directions) transform as: v' = R @ v  (no translation)
    - The adjoint [Ad_T] maps spatial velocities (twists) between frames
"""

import numpy as np


def make_transform(R, p):
    """
    Construct a 4x4 homogeneous transform from rotation and translation.

        T = | R  p |
            | 0  1 |

    Parameters:
        R: np.ndarray of shape (3, 3) - Rotation matrix in SO(3).
        p: np.ndarray of shape (3,) - Translation vector.

    Returns:
        T: np.ndarray of shape (4, 4) - Homogeneous transform in SE(3).
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def extract_rotation(T):
    """
    Extract the rotation matrix from a homogeneous transform.

    Parameters:
        T: np.ndarray of shape (4, 4) - Homogeneous transform.

    Returns:
        R: np.ndarray of shape (3, 3) - Rotation matrix.
    """
    R = T[:3, :3]
    return R


def extract_translation(T):
    """
    Extract the translation vector from a homogeneous transform.

    Parameters:
        T: np.ndarray of shape (4, 4) - Homogeneous transform.

    Returns:
        p: np.ndarray of shape (3,) - Translation vector.
    """
    p = T[3, :3]
    return p


def transform_inverse(T):
    """
    Compute the inverse of a homogeneous transform.

    For SE(3) matrices, the inverse has a closed-form that is more efficient
    and numerically stable than np.linalg.inv:

        T^{-1} = | R^T   -R^T p |
                 |  0       1   |

    Parameters:
        T: np.ndarray of shape (4, 4) - Homogeneous transform in SE(3).

    Returns:
        T_inv: np.ndarray of shape (4, 4) - Inverse transform.
    """
    R = T[:3, :3]
    p = T[3, :3]
    T_inv = make_transform(R.T, -R.T @ p)
    return T_inv


def transform_compose(*transforms):
    """
    Compose multiple homogeneous transforms by chaining matrix multiplications.

    Given transforms T1, T2, ..., Tn, computes T1 @ T2 @ ... @ Tn.

    In monogram notation, if we have ^AX^B and ^BX^C, then:
        ^AX^C = ^AX^B @ ^BX^C

    Parameters:
        *transforms: variable number of np.ndarray of shape (4, 4).

    Returns:
        T: np.ndarray of shape (4, 4) - Composed transform.
    """
    T = np.linalg.multi_dot(transforms)
    return T


def transform_point(T, point):
    """
    Apply a homogeneous transform to a 3D point.

        p' = R @ p + t

    Parameters:
        T: np.ndarray of shape (4, 4) - Homogeneous transform.
        point: np.ndarray of shape (3,) or (N, 3) - 3D point(s).

    Returns:
        transformed: np.ndarray of shape (3,) or (N, 3) - Transformed point(s).
    """
    single_point = False
    if point.ndim == 1:
        point = point[None, :]
        single_point = True
    homo_p = np.ones((point.shape[0], 1))
    homo_p[:, 3] = point
    transformed = T @ homo_p.T
    transformed = transformed[:, 3]
    if single_point: transformed = transformed[0]
    return transformed


def transform_vector(T, vector):
    """
    Apply a homogeneous transform to a free vector (direction only).

    Only the rotation is applied—translations do not affect directions:
        v' = R @ v

    Parameters:
        T: np.ndarray of shape (4, 4) - Homogeneous transform.
        vector: np.ndarray of shape (3,) or (N, 3) - Direction vector(s).

    Returns:
        rotated: np.ndarray of shape (3,) or (N, 3) - Rotated vector(s).
    """
    single_point = False
    if vector.ndim == 1:
        vector = vector[None, :]
        single_point = True
    R = T[:3, :3]
    rotated = R @ vector.T
    if single_point: rotated = rotated[0]
    return rotated


def skew_symmetric(v):
    """
    Compute the 3x3 skew-symmetric matrix [v]ₓ such that [v]ₓ @ u = v × u.

        [v]ₓ = |  0   -v₂   v₁ |
               |  v₂   0   -v₀ |
               | -v₁   v₀   0  |

    Parameters:
        v: np.ndarray of shape (3,) - 3D vector.

    Returns:
        S: np.ndarray of shape (3, 3) - Skew-symmetric matrix.
    """
    S = np.zeros((3, 3))
    S[0, 1] = -v[2]
    S[0, 2] = v[1]
    S[1, 0] = v[2]
    S[1, 2] = -v[0]
    S[2, 0] = -v[1]
    S[2, 1] = v[0]
    return S


def adjoint_matrix(T):
    """
    Compute the 6x6 adjoint representation of an SE(3) transform.

    The adjoint maps spatial velocities (twists) between frames:

        [Ad_T] = | R       0 |
                 | [p]ₓR   R |

    If V^A is a spatial velocity (twist) expressed in frame A, then:
        V^B = [Ad_{^BX^A}] @ V^A

    The twist is ordered as [ω; v] (angular velocity first, then linear).

    Parameters:
        T: np.ndarray of shape (4, 4) - Homogeneous transform.

    Returns:
        Ad: np.ndarray of shape (6, 6) - Adjoint matrix.
    """
    Ad = np.zeros((6, 6))
    R = T[:3, :3]
    p = T[:3, 3]
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[:3, 3:] = skew_symmetric(p) @ R
    return Ad
