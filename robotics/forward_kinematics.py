"""
Forward kinematics for serial manipulators.

Covers DH (Denavit-Hartenberg) parameter convention, planar manipulators,
and general serial chain forward kinematics.

Reference: https://manipulation.csail.mit.edu/pick.html

The forward kinematics problem: given joint configuration q, compute the
pose of the end-effector:
    X^{end-effector} = f_kin(q)

This is done by composing transforms along the kinematic chain:
    T_0^n = T_0^1(q1) @ T_1^2(q2) @ ... @ T_{n-1}^n(qn)
"""

import numpy as np
from .transforms import make_transform
from .rotations import rotation_matrix_z, rotation_matrix_y, rotation_matrix_x


def dh_transform(theta, d, a, alpha):
    """
    Compute the homogeneous transform for one link using standard DH parameters.

    Standard DH convention:
        T = Rz(θ) @ Tz(d) @ Tx(a) @ Rx(α)

    Which expands to:
        T = | cos(θ)  -sin(θ)cos(α)   sin(θ)sin(α)   a·cos(θ) |
            | sin(θ)   cos(θ)cos(α)  -cos(θ)sin(α)   a·sin(θ) |
            |   0       sin(α)         cos(α)          d       |
            |   0         0              0              1       |

    Parameters:
        theta: float - Joint angle (rotation about z-axis) in radians.
        d: float - Link offset (translation along z-axis).
        a: float - Link length (translation along x-axis).
        alpha: float - Link twist (rotation about x-axis) in radians.

    Returns:
        T: np.ndarray of shape (4, 4) - Homogeneous transform for this link.
    """
    T_parent_joint = make_transform(rotation_matrix_z(theta), np.zeros((3)))
    T_joint_link = make_transform(rotation_matrix_x(alpha), np.array([a, 0, d]))
    T_parent_link = T_parent_joint @ T_joint_link
    return T_parent_link


def forward_kinematics_dh(dh_table, joint_angles):
    """
    Compute forward kinematics from a DH parameter table.

    Each row of dh_table is [theta_offset, d, a, alpha]. For revolute joints,
    the actual theta = theta_offset + joint_angle.

    The end-effector pose is:
        T_0^n = T_0^1(q1) @ T_1^2(q2) @ ... @ T_{n-1}^n(qn)

    Parameters:
        dh_table: np.ndarray of shape (n_joints, 4)
            DH parameter table. Columns: [theta_offset, d, a, alpha].
        joint_angles: np.ndarray of shape (n_joints,)
            Joint angles in radians (for revolute joints).

    Returns:
        T: np.ndarray of shape (4, 4) - End-effector pose in the base frame.
    """
    dh_table[:,0] += joint_angles
    forward_transforms = [dh_transform(*args) for args in dh_table]
    if len(forward_transforms) == 1:
        return forward_transforms[0]
    T = np.linalg.multi_dot(forward_transforms)
    return T


def planar_2r_fk(L1, L2, theta1, theta2):
    """
    Forward kinematics for a 2-link planar revolute (2R) arm.

    Joint 1 is at the origin, rotating in the xy-plane.
    Link 1 has length L1, link 2 has length L2.

    Elbow position:
        x1 = L1 cos(θ1)
        y1 = L1 sin(θ1)

    End-effector position:
        x2 = L1 cos(θ1) + L2 cos(θ1 + θ2)
        y2 = L1 sin(θ1) + L2 sin(θ1 + θ2)

    Parameters:
        L1: float - Length of link 1.
        L2: float - Length of link 2.
        theta1: float - Joint 1 angle in radians.
        theta2: float - Joint 2 angle in radians.

    Returns:
        result: dict with keys:
            'joint1': np.ndarray of shape (2,) - Position of joint 1 (origin).
            'joint2': np.ndarray of shape (2,) - Position of joint 2 (elbow).
            'end_effector': np.ndarray of shape (2,) - End-effector position.
            'T_02': np.ndarray of shape (4, 4) - Base-to-end-effector transform.
    """
    T_origin_link1 = dh_transform(theta1, 0, L1, 0)
    T_link1_link2 = dh_transform(theta2, 0, L2, 0)
    T_origin_ee = T_origin_link1 @ T_link1_link2
    result = {
        'joint1': np.zeros((2,)),
        'joint2': T_origin_link1[:2, 3],
        'end_effector': T_origin_ee[:2, 3],
        'T_02': T_origin_ee,
    }
    return result


def planar_3r_fk(L1, L2, L3, theta1, theta2, theta3):
    """
    Forward kinematics for a 3-link planar revolute (3R) arm.

    Each joint accumulates orientation:
        Elbow 1: x1 = L1 cos(θ1), y1 = L1 sin(θ1)
        Elbow 2: x2 = x1 + L2 cos(θ1+θ2), y2 = y1 + L2 sin(θ1+θ2)
        End-eff: x3 = x2 + L3 cos(θ1+θ2+θ3), y3 = y2 + L3 sin(θ1+θ2+θ3)

    Parameters:
        L1, L2, L3: float - Link lengths.
        theta1, theta2, theta3: float - Joint angles in radians.

    Returns:
        result: dict with keys:
            'joint1': np.ndarray of shape (2,) - Origin.
            'joint2': np.ndarray of shape (2,) - Elbow 1 position.
            'joint3': np.ndarray of shape (2,) - Elbow 2 position.
            'end_effector': np.ndarray of shape (2,) - End-effector position.
            'orientation': float - Total end-effector orientation angle.
    """
    T_origin_link1 = dh_transform(theta1, 0, L1, 0)
    T_link1_link2 = dh_transform(theta2, 0, L2, 0)
    T_origin_link2 = T_origin_link1 @ T_link1_link2
    T_link2_link3 = dh_transform(theta3, 0, L3, 0)
    T_origin_ee = T_origin_link2 @ T_link2_link3
    result = {
        'joint1': np.zeros((2,)),
        'joint2': T_origin_link1[:2, 3],
        'joint3':  T_origin_link2[:2, 3],
        'end_effector': T_origin_ee[:2, 3],
        'orientation': np.arctan2(T_origin_ee[1,0],T_origin_ee[0,0]),
    }
    return result
