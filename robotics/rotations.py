"""
Rotation representations and conversions.

Covers SO(3) rotation matrices, Roll-Pitch-Yaw (RPY) Euler angles,
axis-angle (Rodrigues' formula), unit quaternions, and SLERP interpolation.

Reference: https://manipulation.csail.mit.edu/pick.html

Key concepts:
    - SO(3): The group of 3x3 rotation matrices (R^T R = I, det(R) = +1)
    - RPY: Extrinsic XYZ Euler angles: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    - Axis-angle: Rodrigues' formula: R = I cos(θ) + (1-cos(θ))kkᵀ + sin(θ)[k]ₓ
    - Quaternion: q = [w, x, y, z], ||q|| = 1, representing rotation via q v q*
"""

import numpy as np
from numpy.lib.scimath import arccos


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


def rotation_matrix_x(theta):
    """
    Rotation matrix for rotation about the x-axis by angle theta.

    Rx(θ) = | 1    0       0    |
            | 0  cos(θ)  -sin(θ) |
            | 0  sin(θ)   cos(θ) |

    Parameters:
        theta: float - Rotation angle in radians.

    Returns:
        R: np.ndarray of shape (3, 3) - Rotation matrix in SO(3).
    """
    R = np.eye(3)
    R[:, 0] = [1, 0, 0]
    R[:, 1] = [0, np.cos(theta), np.sin(theta)]
    R[:, 2] = [0, -np.sin(theta), np.cos(theta)]
    return R


def rotation_matrix_y(theta):
    """
    Rotation matrix for rotation about the y-axis by angle theta.

    Ry(θ) = |  cos(θ)  0  sin(θ) |
            |    0     1    0    |
            | -sin(θ)  0  cos(θ) |

    Parameters:
        theta: float - Rotation angle in radians.

    Returns:
        R: np.ndarray of shape (3, 3) - Rotation matrix in SO(3).
    """
    R = np.eye(3)
    R[:, 0] = [np.cos(theta), 0, -np.sin(theta)]
    R[:, 1] = [0, 1, 0]
    R[:, 2] = [np.sin(theta), 0, np.cos(theta)]
    return R


def rotation_matrix_z(theta):
    """
    Rotation matrix for rotation about the z-axis by angle theta.

    Rz(θ) = | cos(θ)  -sin(θ)  0 |
            | sin(θ)   cos(θ)  0 |
            |   0        0     1 |

    Parameters:
        theta: float - Rotation angle in radians.

    Returns:
        R: np.ndarray of shape (3, 3) - Rotation matrix in SO(3).
    """
    R = np.eye(3)
    R[:, 0] = [np.cos(theta), np.sin(theta), 0]
    R[:, 1] = [-np.sin(theta), np.cos(theta), 0]
    R[:, 2] = [0, 0, 1]
    return R


def rpy_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Roll-Pitch-Yaw Euler angles to a rotation matrix.

    Uses the extrinsic XYZ convention (rotate about fixed axes):
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Parameters:
        roll: float - Rotation about x-axis in radians.
        pitch: float - Rotation about y-axis in radians.
        yaw: float - Rotation about z-axis in radians.

    Returns:
        R: np.ndarray of shape (3, 3) - Rotation matrix in SO(3).
    """
    R = rotation_matrix_z(yaw) @ rotation_matrix_y(pitch) @ rotation_matrix_x(roll)
    return R


def rotation_matrix_to_rpy(R):
    """
    Extract Roll-Pitch-Yaw angles from a rotation matrix.

    Assumes extrinsic XYZ convention. Gimbal lock occurs at pitch = ±π/2.

    Extraction formulas (non-singular case):
        pitch = arcsin(-R[2,0])
        roll  = arctan2(R[2,1], R[2,2])
        yaw   = arctan2(R[1,0], R[0,0])

    Parameters:
        R: np.ndarray of shape (3, 3) - Rotation matrix in SO(3).

    Returns:
        roll: float - Rotation about x-axis in radians.
        pitch: float - Rotation about y-axis in radians.
        yaw: float - Rotation about z-axis in radians.
    """
    roll = np.arctan2(R[2,1], R[2,2])
    pitch = np.arcsin(-R[2,0])
    yaw = np.arctan2(R[1,0], R[0,0])
    return roll, pitch, yaw


def axis_angle_to_rotation_matrix(axis, angle):
    """
    Convert axis-angle representation to rotation matrix using Rodrigues' formula.

    R = I cos(θ) + (1 - cos(θ)) k kᵀ + sin(θ) [k]ₓ

    where [k]ₓ is the skew-symmetric matrix of the unit axis k:
        [k]ₓ = |  0   -k₂   k₁ |
               |  k₂   0   -k₀ |
               | -k₁   k₀   0  |

    Parameters:
        axis: np.ndarray of shape (3,) - Rotation axis (will be normalized).
        angle: float - Rotation angle in radians.

    Returns:
        R: np.ndarray of shape (3, 3) - Rotation matrix in SO(3).
    """
    axis = axis / np.linalg.norm(axis)
    R = (
        np.cos(angle) * np.eye(3)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * skew_symmetric(axis)
    )
    return R


def rotation_matrix_to_axis_angle(R):
    """
    Extract axis-angle representation from a rotation matrix.

    General case:
        angle = arccos((trace(R) - 1) / 2)
        axis  = (1 / 2sin(θ)) * [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]

    Special cases:
        - angle ≈ 0: axis is arbitrary (no rotation).
        - angle ≈ π: extract axis from R + I (pick column with largest norm).

    Parameters:
        R: np.ndarray of shape (3, 3) - Rotation matrix in SO(3).

    Returns:
        axis: np.ndarray of shape (3,) - Unit rotation axis.
        angle: float - Rotation angle in radians in [0, π].
    """
    angle = np.arccos((np.trace(R) - 1) / 2)
    axis = (1/(2 * np.sin(angle))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return axis, angle


def quaternion_to_rotation_matrix(q):
    """
    Convert unit quaternion to rotation matrix.

    Quaternion convention: q = [w, x, y, z] where w is the scalar part.

    R = | 1-2(y²+z²)   2(xy-wz)    2(xz+wy)  |
        | 2(xy+wz)     1-2(x²+z²)  2(yz-wx)  |
        | 2(xz-wy)     2(yz+wx)    1-2(x²+y²) |

    Parameters:
        q: np.ndarray of shape (4,) - Unit quaternion [w, x, y, z].

    Returns:
        R: np.ndarray of shape (3, 3) - Rotation matrix in SO(3).
    """
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2),  2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),        1 - 2*(x**2 + z**2),  2*(y*z - w*x)],
        [2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x**2 + y**2)],
    ])
    return R


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to unit quaternion using Shepperd's method.

    Shepperd's method selects the numerically best branch based on
    which of {trace, R[0,0], R[1,1], R[2,2]} is largest:

    Case 1 (trace > 0):
        s = 2√(trace+1), w = s/4, x = (R[2,1]-R[1,2])/s, etc.

    Case 2 (R[0,0] largest):
        s = 2√(1+R[0,0]-R[1,1]-R[2,2]), x = s/4, w = (R[2,1]-R[1,2])/s, etc.

    (Similar for R[1,1] and R[2,2] largest.)

    Convention: return q with w >= 0.

    Parameters:
        R: np.ndarray of shape (3, 3) - Rotation matrix in SO(3).

    Returns:
        q: np.ndarray of shape (4,) - Unit quaternion [w, x, y, z] with w >= 0.
    """
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (Hamilton product).

    Given q1 = [w1, x1, y1, z1] and q2 = [w2, x2, y2, z2]:

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    Parameters:
        q1: np.ndarray of shape (4,) - First quaternion [w, x, y, z].
        q2: np.ndarray of shape (4,) - Second quaternion [w, x, y, z].

    Returns:
        q: np.ndarray of shape (4,) - Product quaternion q1 * q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    return q


def quaternion_conjugate(q):
    """
    Compute the conjugate of a quaternion.

    q* = [w, -x, -y, -z]

    For unit quaternions, the conjugate equals the inverse:
        q * q* = [1, 0, 0, 0]

    Parameters:
        q: np.ndarray of shape (4,) - Quaternion [w, x, y, z].

    Returns:
        q_conj: np.ndarray of shape (4,) - Conjugate [w, -x, -y, -z].
    """
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return q_conj


def slerp(q0, q1, t):
    """
    Spherical linear interpolation between two unit quaternions.

    SLERP formula:
        slerp(q0, q1, t) = sin((1-t)θ)/sin(θ) * q0 + sin(tθ)/sin(θ) * q1

    where θ = arccos(q0 · q1).

    Special cases:
        - If q0 · q1 < 0, negate q1 to take the shorter path.
        - If q0 ≈ q1 (dot > 0.9995), use linear interpolation.

    Parameters:
        q0: np.ndarray of shape (4,) - Start quaternion [w, x, y, z].
        q1: np.ndarray of shape (4,) - End quaternion [w, x, y, z].
        t: float - Interpolation parameter in [0, 1].

    Returns:
        q: np.ndarray of shape (4,) - Interpolated unit quaternion.
    """
    if np.dot(q0, q1) < 0: q1 = -q1
    if np.dot(q0, q1) > 0.995:
        q = t * q0 + (1 - t) * q1
    else:
        theta = arccos(np.dot(q0,q1))
        q = np.sin((1-t) * theta)/np.sin(theta) * q0 + np.sin((t * theta))/np.sin(theta) * q1
    return q


def is_valid_rotation_matrix(R, tol=1e-6):
    """
    Check whether a matrix is a valid rotation matrix (element of SO(3)).

    Conditions:
        1. R^T R = I  (orthogonal)
        2. det(R) = +1 (proper rotation, not reflection)

    Parameters:
        R: np.ndarray of shape (3, 3) - Matrix to check.
        tol: float - Tolerance for numerical checks.

    Returns:
        valid: bool - True if R is in SO(3).
    """
    if R.shape != (3, 3):
        return False
    valid = (np.allclose(R @ R.T, np.eye(3), tol)) and (np.linalg.det(R) > 1 - tol)
    valid = bool(valid)
    return valid
