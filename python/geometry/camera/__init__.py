"""
Camera Models Module.

Implements camera models for 3D computer vision, including the pinhole camera
model, lens distortion, and camera calibration.

Theory:
    The pinhole camera model projects 3D world points to 2D image coordinates:

    Projection: x = K[R|t]X

    Where:
    - X = (X, Y, Z, 1)^T is a 3D point in homogeneous coordinates
    - K is the 3x3 intrinsic matrix (focal length, principal point)
    - [R|t] is the 3x4 extrinsic matrix (rotation and translation)
    - x = (u, v, 1)^T is the 2D image point

    Intrinsic Matrix:
        K = [f_x  s   c_x]
            [0    f_y c_y]
            [0    0   1  ]

    Where f_x, f_y are focal lengths, (c_x, c_y) is principal point, s is skew.

References:
    - "Multiple View Geometry" Chapter 6 (Hartley & Zisserman)
    - "A Flexible New Technique for Camera Calibration" (Zhang, 2000)
      https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/

Implementation Status: STUB
Complexity: Intermediate
Prerequisites: foundations (linear algebra)
"""

import numpy as np
from typing import Tuple, Optional, Union

__all__ = ['PinholeCamera', 'CameraIntrinsics', 'CameraDistortion']


class CameraIntrinsics:
    """
    Camera intrinsic parameters.

    Theory:
        Intrinsics describe the internal parameters of a camera:
        - Focal length (f_x, f_y): How the camera maps 3D to 2D scale
        - Principal point (c_x, c_y): Where the optical axis hits the image
        - Skew (s): Non-perpendicularity of pixel axes (usually 0)

    Math:
        Intrinsic matrix K:
            K = [f_x  s   c_x]
                [0    f_y c_y]
                [0    0   1  ]

    Example:
        >>> K = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        >>> matrix = K.get_matrix()
        >>> K_inv = K.get_inverse()
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        skew: float = 0.0
    ):
        """
        Initialize camera intrinsics.

        Args:
            fx: Focal length in x (pixels)
            fy: Focal length in y (pixels)
            cx: Principal point x coordinate
            cy: Principal point y coordinate
            skew: Skew coefficient (usually 0)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.skew = skew

    def get_matrix(self) -> np.ndarray:
        """
        Get 3x3 intrinsic matrix K.

        Implementation:
            Return np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]])
        """
        raise NotImplementedError(
            "Build intrinsic matrix. "
            "K = [[fx, s, cx], [0, fy, cy], [0, 0, 1]]"
        )

    def get_inverse(self) -> np.ndarray:
        """
        Get inverse of intrinsic matrix K^-1.

        Implementation hints:
            K^-1 = [[1/fx, -s/(fx*fy), (s*cy-cx*fy)/(fx*fy)],
                    [0,    1/fy,       -cy/fy],
                    [0,    0,          1]]
        """
        raise NotImplementedError(
            "Compute inverse intrinsic matrix. "
            "Use closed-form inverse for upper triangular matrix."
        )

    def pixel_to_normalized(self, pixels: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates to normalized camera coordinates.

        Args:
            pixels: (N, 2) array of pixel coordinates (u, v)

        Returns:
            (N, 2) array of normalized coordinates
        """
        raise NotImplementedError(
            "Apply K^-1 to convert pixels to normalized coords. "
            "x_norm = (u - cx) / fx, y_norm = (v - cy) / fy"
        )

    def normalized_to_pixel(self, normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized camera coordinates to pixels.

        Args:
            normalized: (N, 2) array of normalized coordinates

        Returns:
            (N, 2) array of pixel coordinates
        """
        raise NotImplementedError(
            "Apply K to convert normalized coords to pixels. "
            "u = fx * x + cx, v = fy * y + cy"
        )


class CameraDistortion:
    """
    Lens distortion model.

    Theory:
        Real lenses introduce distortion, primarily radial and tangential.
        We model this as a function that maps ideal (undistorted) points
        to observed (distorted) points.

    Radial Distortion:
        x_d = x(1 + k1*r² + k2*r⁴ + k3*r⁶)
        y_d = y(1 + k1*r² + k2*r⁴ + k3*r⁶)

        Where r² = x² + y² and k1, k2, k3 are radial coefficients.

    Tangential Distortion:
        x_d += 2*p1*x*y + p2*(r² + 2*x²)
        y_d += p1*(r² + 2*y²) + 2*p2*x*y

        Where p1, p2 are tangential coefficients.

    References:
        - OpenCV camera calibration documentation
        - "A Flexible New Technique for Camera Calibration" (Zhang, 2000)
    """

    def __init__(
        self,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0
    ):
        """
        Initialize distortion model.

        Args:
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
        """
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2

    def distort(self, points: np.ndarray) -> np.ndarray:
        """
        Apply distortion to normalized points.

        Args:
            points: (N, 2) undistorted normalized coordinates

        Returns:
            (N, 2) distorted normalized coordinates

        Implementation hints:
            1. Compute r² = x² + y²
            2. Radial factor: (1 + k1*r² + k2*r⁴ + k3*r⁶)
            3. Apply radial and tangential distortion
        """
        raise NotImplementedError(
            "Apply distortion model. "
            "Compute radial and tangential distortion."
        )

    def undistort(
        self,
        points: np.ndarray,
        iterations: int = 10
    ) -> np.ndarray:
        """
        Remove distortion from points (iterative method).

        Args:
            points: (N, 2) distorted normalized coordinates
            iterations: Number of iterations for refinement

        Returns:
            (N, 2) undistorted normalized coordinates

        Implementation hints:
            Iterative undistortion:
            1. Initialize: x_und = x_dist
            2. For each iteration:
               a. Compute what x_und would distort to
               b. Compute error = x_dist - distort(x_und)
               c. Update: x_und += error
        """
        raise NotImplementedError(
            "Iterative undistortion. "
            "Refine estimate by computing distortion error."
        )


class PinholeCamera:
    """
    Complete pinhole camera model with extrinsics.

    Theory:
        The full projection from world to image is:
            x = K[R|t]X

        Where:
        - X is a 3D world point
        - [R|t] transforms from world to camera frame
        - K projects from camera frame to image

    Coordinate Frames:
        - World: Global reference frame
        - Camera: Origin at camera center, Z forward, Y down
        - Image: Origin at top-left, u right, v down

    Example:
        >>> intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        >>> camera = PinholeCamera(intrinsics, R, t)
        >>> pixels = camera.project(world_points)
        >>> rays = camera.unproject(pixels)
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        R: np.ndarray = None,
        t: np.ndarray = None,
        distortion: CameraDistortion = None
    ):
        """
        Initialize pinhole camera.

        Args:
            intrinsics: Camera intrinsic parameters
            R: 3x3 rotation matrix (world to camera)
            t: 3x1 translation vector (world to camera)
            distortion: Optional lens distortion model
        """
        self.intrinsics = intrinsics
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros((3, 1))
        self.distortion = distortion

    def get_projection_matrix(self) -> np.ndarray:
        """
        Get 3x4 projection matrix P = K[R|t].

        Returns:
            3x4 projection matrix
        """
        raise NotImplementedError(
            "Compute P = K @ [R|t]. "
            "Concatenate R and t, then multiply by K."
        )

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D world points to 2D image coordinates.

        Args:
            points_3d: (N, 3) array of 3D world points

        Returns:
            (N, 2) array of 2D image coordinates

        Implementation hints:
            1. Transform to camera frame: X_cam = R @ X_world + t
            2. Perspective divide: x_norm = X_cam[:2] / X_cam[2]
            3. Apply distortion if present
            4. Apply intrinsics: pixels = K @ x_norm
        """
        raise NotImplementedError(
            "Implement full projection pipeline. "
            "World -> Camera -> Normalized -> Distorted -> Pixels"
        )

    def unproject(
        self,
        pixels: np.ndarray,
        depth: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Unproject 2D pixels to 3D rays or points.

        Args:
            pixels: (N, 2) array of pixel coordinates
            depth: Optional (N,) array of depths. If None, returns unit rays.

        Returns:
            (N, 3) array of 3D points (if depth given) or ray directions
        """
        raise NotImplementedError(
            "Implement unprojection. "
            "Pixels -> Normalized -> Undistorted -> Camera rays -> World rays"
        )

    def get_camera_center(self) -> np.ndarray:
        """
        Get camera center in world coordinates.

        The camera center C satisfies: t = -R @ C
        Therefore: C = -R^T @ t

        Returns:
            (3,) camera center in world frame
        """
        raise NotImplementedError(
            "Compute camera center. "
            "C = -R.T @ t"
        )

    def get_optical_axis(self) -> np.ndarray:
        """
        Get optical axis (viewing direction) in world coordinates.

        Returns:
            (3,) unit vector pointing in camera's viewing direction
        """
        raise NotImplementedError(
            "Get viewing direction. "
            "Third row of R (camera Z in world coords)"
        )


# Utility functions

def rotation_matrix_from_angles(
    roll: float,
    pitch: float,
    yaw: float
) -> np.ndarray:
    """
    Create rotation matrix from Euler angles.

    Args:
        roll: Rotation around X axis (radians)
        pitch: Rotation around Y axis (radians)
        yaw: Rotation around Z axis (radians)

    Returns:
        3x3 rotation matrix
    """
    raise NotImplementedError(
        "Compute R = Rz(yaw) @ Ry(pitch) @ Rx(roll). "
        "Use basic rotation matrices around each axis."
    )


def calibrate_camera(
    object_points: np.ndarray,
    image_points: np.ndarray,
    image_size: Tuple[int, int]
) -> Tuple[CameraIntrinsics, CameraDistortion]:
    """
    Calibrate camera from 2D-3D correspondences.

    Uses Zhang's method for camera calibration.

    Args:
        object_points: (N, 3) 3D points (e.g., checkerboard corners)
        image_points: (N, 2) corresponding 2D image points
        image_size: (width, height) of images

    Returns:
        Estimated intrinsics and distortion parameters
    """
    raise NotImplementedError(
        "Implement Zhang's calibration method. "
        "Compute homographies, then solve for intrinsics."
    )
