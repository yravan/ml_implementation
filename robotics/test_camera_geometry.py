import numpy as np
import pytest
from robotics.camera_geometry import (
    intrinsic_matrix, pinhole_projection, project_world_to_image,
    pixel_to_ray, pixel_to_3d_point, depth_to_point_cloud,
)
from robotics.transforms import make_transform
from robotics.rotations import rotation_matrix_x, rotation_matrix_y


class TestIntrinsicMatrix:
    """Tests for camera intrinsic matrix construction."""

    def test_shape(self):
        K = intrinsic_matrix(500, 500, 320, 240)
        assert K.shape == (3, 3)

    def test_values(self):
        K = intrinsic_matrix(500, 600, 320, 240)
        expected = np.array([
            [500, 0,   320],
            [0,   600, 240],
            [0,   0,   1  ],
        ])
        np.testing.assert_array_almost_equal(K, expected)

    def test_bottom_row(self):
        K = intrinsic_matrix(100, 200, 50, 50)
        np.testing.assert_array_almost_equal(K[2, :], [0, 0, 1])


class TestPinholeProjection:
    """Tests for pinhole camera projection."""

    def test_on_axis_point(self):
        """Point on the z-axis should project to the principal point."""
        K = intrinsic_matrix(500, 500, 320, 240)
        point = np.array([0, 0, 5])
        pixel = pinhole_projection(K, point)
        np.testing.assert_array_almost_equal(pixel, [320, 240])

    def test_known_projection(self):
        """Verify with manually computed projection."""
        K = intrinsic_matrix(500, 500, 320, 240)
        # Point at (1, 0, 5) -> u = 500*1/5 + 320 = 420, v = 240
        point = np.array([1.0, 0, 5.0])
        pixel = pinhole_projection(K, point)
        np.testing.assert_array_almost_equal(pixel, [420, 240])

    def test_depth_scaling(self):
        """Doubling depth should move pixel closer to principal point."""
        K = intrinsic_matrix(500, 500, 320, 240)
        p1 = np.array([2.0, 1.0, 5.0])
        p2 = np.array([2.0, 1.0, 10.0])
        px1 = pinhole_projection(K, p1)
        px2 = pinhole_projection(K, p2)
        # px2 should be closer to (320, 240) than px1
        d1 = np.linalg.norm(px1 - np.array([320, 240]))
        d2 = np.linalg.norm(px2 - np.array([320, 240]))
        assert d2 < d1

    def test_batch_projection(self):
        """Batch of points should project correctly."""
        K = intrinsic_matrix(500, 500, 320, 240)
        points = np.array([
            [0, 0, 5],
            [1, 0, 5],
            [0, 1, 5],
        ], dtype=float)
        pixels = pinhole_projection(K, points)
        assert pixels.shape == (3, 2)
        np.testing.assert_array_almost_equal(pixels[0], [320, 240])


class TestProjectWorldToImage:
    """Tests for full world-to-pixel projection pipeline."""

    def test_identity_extrinsic(self):
        """With identity extrinsic, should match direct projection."""
        K = intrinsic_matrix(500, 500, 320, 240)
        T = np.eye(4)
        point = np.array([1.0, 0.5, 5.0])
        px_direct = pinhole_projection(K, point)
        px_world = project_world_to_image(K, T, point)
        np.testing.assert_array_almost_equal(px_world, px_direct)

    def test_translated_camera(self):
        """Camera shifted along x should shift projection."""
        K = intrinsic_matrix(500, 500, 320, 240)
        # Camera at world position (1, 0, 0), looking along z
        # T_camera_world moves world points into camera frame
        # If camera is at (1,0,0) in world, point at (1,0,5) in world
        # is at (0,0,5) in camera frame
        T = make_transform(np.eye(3), np.array([-1, 0, 0]))  # shift world by -1 in x
        point_world = np.array([1.0, 0, 5.0])
        pixel = project_world_to_image(K, T, point_world)
        # In camera frame: (0, 0, 5) -> principal point
        np.testing.assert_array_almost_equal(pixel, [320, 240])

    def test_batch_projection(self):
        """Batch of world points should project correctly."""
        K = intrinsic_matrix(500, 500, 320, 240)
        T = np.eye(4)
        points = np.array([
            [0, 0, 5],
            [1, 0, 5],
        ], dtype=float)
        pixels = project_world_to_image(K, T, points)
        assert pixels.shape == (2, 2)


class TestPixelToRay:
    """Tests for back-projection to rays."""

    def test_principal_point_ray(self):
        """Principal point should back-project to [0, 0, 1] direction."""
        K = intrinsic_matrix(500, 500, 320, 240)
        ray = pixel_to_ray(K, np.array([320, 240]))
        expected = np.array([0, 0, 1])
        np.testing.assert_array_almost_equal(ray, expected)

    def test_unit_length(self):
        """Output ray should be unit length."""
        K = intrinsic_matrix(500, 500, 320, 240)
        ray = pixel_to_ray(K, np.array([100, 200]))
        np.testing.assert_almost_equal(np.linalg.norm(ray), 1.0)

    def test_positive_z(self):
        """Ray z-component should be positive (pointing forward)."""
        K = intrinsic_matrix(500, 500, 320, 240)
        for u in [0, 320, 640]:
            for v in [0, 240, 480]:
                ray = pixel_to_ray(K, np.array([u, v]))
                assert ray[2] > 0

    def test_consistency_with_projection(self):
        """Projecting a point on the ray should recover the original pixel."""
        K = intrinsic_matrix(500, 500, 320, 240)
        pixel_orig = np.array([400.0, 300.0])
        ray = pixel_to_ray(K, pixel_orig)
        # Scale ray to some depth
        point_3d = ray * 10.0 / ray[2]  # depth = 10
        pixel_back = pinhole_projection(K, point_3d)
        np.testing.assert_array_almost_equal(pixel_back, pixel_orig, decimal=5)


class TestPixelTo3DPoint:
    """Tests for back-projection with known depth."""

    def test_principal_point(self):
        """Principal point at depth Z should give (0, 0, Z)."""
        K = intrinsic_matrix(500, 500, 320, 240)
        point = pixel_to_3d_point(K, np.array([320, 240]), 5.0)
        np.testing.assert_array_almost_equal(point, [0, 0, 5])

    def test_roundtrip_with_projection(self):
        """project(back_project(pixel, Z)) should recover pixel."""
        K = intrinsic_matrix(500, 500, 320, 240)
        pixel_orig = np.array([150.0, 350.0])
        depth = 8.0
        point_3d = pixel_to_3d_point(K, pixel_orig, depth)
        pixel_back = pinhole_projection(K, point_3d)
        np.testing.assert_array_almost_equal(pixel_back, pixel_orig, decimal=5)

    def test_depth_preserved(self):
        """Z coordinate of result should equal the input depth."""
        K = intrinsic_matrix(500, 500, 320, 240)
        point = pixel_to_3d_point(K, np.array([100, 200]), 7.5)
        np.testing.assert_almost_equal(point[2], 7.5)


class TestDepthToPointCloud:
    """Tests for depth image to point cloud conversion."""

    def test_single_valid_pixel(self):
        """A 1x1 depth image with valid depth should give one point."""
        K = intrinsic_matrix(500, 500, 0, 0)  # Principal point at origin
        depth_image = np.array([[5.0]])
        points = depth_to_point_cloud(K, depth_image)
        assert points.shape == (1, 3)
        np.testing.assert_array_almost_equal(points[0], [0, 0, 5])

    def test_invalid_depth_excluded(self):
        """Pixels with depth <= 0 should be excluded."""
        K = intrinsic_matrix(500, 500, 1, 1)
        depth_image = np.array([
            [5.0, 0.0],
            [-1.0, 3.0],
        ])
        points = depth_to_point_cloud(K, depth_image)
        assert points.shape[0] == 2  # Only 2 valid pixels

    def test_all_invalid(self):
        """All-zero depth image should give empty point cloud."""
        K = intrinsic_matrix(500, 500, 320, 240)
        depth_image = np.zeros((10, 10))
        points = depth_to_point_cloud(K, depth_image)
        assert points.shape[0] == 0

    def test_z_values_match_depth(self):
        """Z coordinates of points should match depth values."""
        K = intrinsic_matrix(500, 500, 2, 2)
        depth_image = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        points = depth_to_point_cloud(K, depth_image)
        # All pixels valid
        assert points.shape[0] == 6
        # Z values should contain all depth values
        z_values = sorted(points[:, 2])
        np.testing.assert_array_almost_equal(z_values, [1, 2, 3, 4, 5, 6])

    def test_principal_point_projects_to_z_axis(self):
        """Pixel at the principal point should give X=0, Y=0."""
        cx, cy = 2, 1  # Principal point at pixel (2, 1) (col=2, row=1)
        K = intrinsic_matrix(500, 500, cx, cy)
        depth_image = np.zeros((3, 5))
        depth_image[1, 2] = 10.0  # Only valid pixel at (row=1, col=2)
        points = depth_to_point_cloud(K, depth_image)
        assert points.shape[0] == 1
        np.testing.assert_almost_equal(points[0, 0], 0.0)  # X = 0
        np.testing.assert_almost_equal(points[0, 1], 0.0)  # Y = 0
        np.testing.assert_almost_equal(points[0, 2], 10.0)  # Z = depth


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
