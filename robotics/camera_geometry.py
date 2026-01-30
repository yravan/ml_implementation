"""
Camera geometry and projection models.

Covers the pinhole camera model, intrinsic/extrinsic parameters,
projection and back-projection, and depth image to point cloud conversion.

Reference: https://manipulation.csail.mit.edu/pick.html

Key concepts:
    - Intrinsic matrix K encodes focal length and principal point:
        K = | fx  0  cx |
            |  0  fy cy |
            |  0   0  1 |

    - Pinhole projection: pixel = (1/Z) K @ p_camera
    - Full pipeline: world -> camera frame (extrinsic T) -> pixel (intrinsic K)
    - Back-projection: pixel + depth -> 3D point in camera frame
"""

import numpy as np


def intrinsic_matrix(fx, fy, cx, cy):
    """
    Construct the 3x3 camera intrinsic matrix.

        K = | fx  0  cx |
            |  0  fy cy |
            |  0   0  1 |

    Parameters:
        fx: float - Focal length in x (pixels).
        fy: float - Focal length in y (pixels).
        cx: float - Principal point x-coordinate (pixels).
        cy: float - Principal point y-coordinate (pixels).

    Returns:
        K: np.ndarray of shape (3, 3) - Camera intrinsic matrix.
    """
    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def pinhole_projection(K, point_camera):
    """
    Project a 3D point (in camera frame) to a 2D pixel using the pinhole model.

    The projection is:
        p_homogeneous = K @ p_camera   (gives [u*Z, v*Z, Z])
        pixel = p_homogeneous[:2] / p_homogeneous[2]   (gives [u, v])

    Parameters:
        K: np.ndarray of shape (3, 3) - Camera intrinsic matrix.
        point_camera: np.ndarray of shape (3,) or (N, 3)
            3D point(s) in camera frame. Z must be > 0.

    Returns:
        pixel: np.ndarray of shape (2,) or (N, 2) - Pixel coordinates [u, v].
    """
    single_point = False
    if point_camera.ndim == 1:
        point_camera = point_camera[np.newaxis, :]
        single_point = True
    p_homogeneous = (K @ point_camera.T).T
    pixel = p_homogeneous[...,:2] / p_homogeneous[...,2:]
    if single_point: pixel = pixel[0]
    return pixel


def project_world_to_image(K, T_camera_world, point_world):
    """
    Project a 3D world-frame point to pixel coordinates.

    Full pipeline: world frame -> camera frame -> pixel.

    In monogram notation:
        ^Cp^P = ^CX^W @ ^Wp^P   (transform point to camera frame)
        pixel = project(K, ^Cp^P)

    Parameters:
        K: np.ndarray of shape (3, 3) - Camera intrinsic matrix.
        T_camera_world: np.ndarray of shape (4, 4)
            Extrinsic transform from world to camera frame (^CX^W).
        point_world: np.ndarray of shape (3,) or (N, 3) - Point(s) in world frame.

    Returns:
        pixel: np.ndarray of shape (2,) or (N, 2) - Pixel coordinates [u, v].
    """
    single_point = False
    if point_world.ndim == 1:
        point_world = point_world[np.newaxis, :]
        single_point = True
    point_world_homogeneous = np.ones((point_world.shape[0], 4))
    point_world_homogeneous[...,:3] = point_world
    point_camera = (T_camera_world @ point_world_homogeneous.T).T
    pixel = pinhole_projection(K, point_camera[:, :3])
    if single_point: pixel = pixel[0]
    return pixel


def pixel_to_ray(K, pixel):
    """
    Back-project a pixel to a 3D ray in the camera frame.

    The ray direction is K⁻¹ @ [u, v, 1]ᵀ, normalized to unit length.

    This gives the direction along which the 3D point lies, but not the depth.

    Parameters:
        K: np.ndarray of shape (3, 3) - Camera intrinsic matrix.
        pixel: np.ndarray of shape (2,) - Pixel coordinates [u, v].

    Returns:
        ray: np.ndarray of shape (3,) - Unit ray direction in camera frame.
    """
    single_point = False
    if pixel.ndim == 1:
        pixel = pixel[np.newaxis, :]
        single_point = True
    pixel_homogeneous = np.ones((pixel.shape[0], 3))
    pixel_homogeneous[...,0] = pixel[:, 0]
    pixel_homogeneous[...,1] = pixel[:, 1]
    f_x, f_y, c_x, c_y = K[0,0], K[1,1], K[0,2], K[1,2]
    K_inv = intrinsic_matrix(1/f_x, 1/f_y, -c_x/f_x, -c_y/f_y)
    ray = (K_inv @ pixel_homogeneous.T).T
    if single_point:
        ray = ray[0]
    return ray / np.linalg.norm(ray)


def pixel_to_3d_point(K, pixel, depth):
    """
    Back-project a pixel with known depth to a 3D point in camera frame.

    Given pixel [u, v] and depth Z:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    Parameters:
        K: np.ndarray of shape (3, 3) - Camera intrinsic matrix.
        pixel: np.ndarray of shape (2,) - Pixel coordinates [u, v].
        depth: float - Depth value (Z in camera frame).

    Returns:
        point: np.ndarray of shape (3,) - 3D point in camera frame.
    """
    ray = pixel_to_ray(K, pixel)
    point = (ray * depth[:,None]) / ray[:, 2:]
    return point


def depth_to_point_cloud(K, depth_image):
    """
    Convert a depth image to a 3D point cloud in the camera frame.

    For each valid pixel (u, v) with depth Z > 0:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        point = [X, Y, Z]

    Parameters:
        K: np.ndarray of shape (3, 3) - Camera intrinsic matrix.
        depth_image: np.ndarray of shape (H, W)
            Depth image where each pixel stores the Z-depth.
            Pixels with depth <= 0 are treated as invalid.

    Returns:
        points: np.ndarray of shape (N, 3) - 3D point cloud (valid pixels only).
    """
    pixel_coords = np.indices(depth_image.shape)
    pixel_coords = pixel_coords.reshape(2, -1).T
    depth_image = depth_image.flatten()
    points = pixel_to_3d_point(K, pixel_coords[depth_image > 0], depth_image[depth_image > 0])
    return points
