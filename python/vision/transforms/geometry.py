"""
Geometry Transforms
===================

Spatial transformations for images.
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Sequence
import cv2


class Resize:
    """
    Resize image to given size.

    Args:
        size: Target size (height, width) or single int for shorter edge
        interpolation: Interpolation mode ('nearest', 'bilinear', 'bicubic')
        max_size: Maximum size of longer edge
        antialias: Whether to use antialiasing
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = 'bilinear',
        max_size: Optional[int] = None,
        antialias: bool = True,
    ):
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert img.ndim == 3, f"Wrong image dimension {img.shape}"
        C, H, W = img.shape

        # Compute target size
        if isinstance(self.size, int):
            if H < W:
                H_new = self.size
                W_new = int(W * self.size / H + 0.5)
            else:
                W_new = self.size
                H_new = int(H * self.size / W + 0.5)

            if self.max_size is not None and max(H_new, W_new) > self.max_size:
                if H_new > W_new:
                    W_new = int(W_new * self.max_size / H_new + 0.5)
                    H_new = self.max_size
                else:
                    H_new = int(H_new * self.max_size / W_new + 0.5)
                    W_new = self.max_size
        else:
            H_new, W_new = self.size

        if H == H_new and W == W_new:
            return img

        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
        }
        # CHW -> HWC for OpenCV
        hwc = img.transpose(1, 2, 0)
        resized = cv2.resize(
            hwc, (W_new, H_new), interpolation=interp_map[self.interpolation]
        )
        return resized.transpose(2, 0, 1)


class RandomResize:
    """Randomly resize image within size range."""

    def __init__(
        self,
        min_size: int,
        max_size: int,
        interpolation: str = 'bilinear',
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert img.ndim == 3, f"Wrong image dimension {img.shape}"
        C, H, W = img.shape

        # Compute target size
        new_size = np.random.randint(self.min_size, self.max_size + 1)
        if H < W:
            H_new = new_size
            W_new = int(W * new_size / H + 0.5)
        else:
            W_new = new_size
            H_new = int(H * new_size / W + 0.5)

        if H == H_new and W == W_new:
            return img

        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
        }
        # CHW -> HWC for OpenCV
        hwc = img.transpose(1, 2, 0)
        resized = cv2.resize(
            hwc, (W_new, H_new), interpolation=interp_map[self.interpolation]
        )
        return resized.transpose(2, 0, 1)



class CenterCrop:
    """
    Crop the center of the image.

    Args:
        size: Target crop size (height, width)
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert img.ndim == 3, f"Wrong image dimension {img.shape}"
        C, H, W = img.shape
        th, tw = self.size
        top = (H - th) // 2
        left = (W - tw) // 2
        return img[:, top:top + th, left:left + tw]


class RandomCrop:
    """
    Randomly crop the image.

    Args:
        size: Target crop size
        padding: Padding before crop
        pad_if_needed: Pad if image smaller than crop size
        fill: Fill value for padding
        padding_mode: Padding mode ('constant', 'edge', 'reflect', 'symmetric')
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        padding: Optional[Union[int, Tuple[int, ...]]] = None,
        pad_if_needed: bool = False,
        fill: Union[int, Tuple[int, ...]] = 0,
        padding_mode: str = 'constant',
    ):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def _pad(self, img, pad_width):
        """Apply padding with the configured mode and fill value."""
        if self.padding_mode == "constant":
            fill = self.fill if isinstance(self.fill, (int, float)) else 0
            return np.pad(img, pad_width, mode="constant", constant_values=fill)
        else:
            return np.pad(img, pad_width, mode=self.padding_mode)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.ndim != 3:
            raise ValueError("Wrong image dimension", img.shape)

        C, H, W = img.shape
        th, tw = self.size

        if self.padding is not None:
            if isinstance(self.padding, int):
                pad_width = (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                )
            elif len(self.padding) == 2:
                pad_width = (
                    (0, 0),
                    (self.padding[1], self.padding[1]),
                    (self.padding[0], self.padding[0]),
                )
            elif len(self.padding) == 4:
                pad_width = (
                    (0, 0),
                    (self.padding[1], self.padding[3]),
                    (self.padding[0], self.padding[2]),
                )
            else:
                raise ValueError(f"Invalid padding: {self.padding}")
            img = self._pad(img, pad_width)
            C, H, W = img.shape

        if self.pad_if_needed and H < th:
            img = self._pad(img, ((0, 0), (0, th - H), (0, 0)))
            C, H, W = img.shape

        if self.pad_if_needed and W < tw:
            img = self._pad(img, ((0, 0), (0, 0), (0, tw - W)))
            C, H, W = img.shape

        if H < th or W < tw:
            raise ValueError(
                f"Image size ({H}, {W}) smaller than crop size ({th}, {tw})"
            )

        top = np.random.randint(0, H - th + 1)
        left = np.random.randint(0, W - tw + 1)

        return img[:, top : top + th, left : left + tw]


class RandomResizedCrop:
    """
    Crop random portion and resize to target size.
    Standard augmentation for ImageNet training.

    Args:
        size: Target size
        scale: Range of crop area ratio (min, max)
        ratio: Range of aspect ratio (min, max)
        interpolation: Interpolation mode
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: str = 'bilinear',
    ):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.resize = Resize(self.size, self.interpolation)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert img.ndim == 3, f"Wrong image dimension {img.shape}"
        C, H, W = img.shape
        area = H * W

        import cv2
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
        }

        # Try up to 10 times to find a valid crop (matches PyTorch)
        for _ in range(10):
            target_area = area * np.random.uniform(self.scale[0], self.scale[1])
            log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))
            aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

            crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < crop_w <= W and 0 < crop_h <= H:
                top = np.random.randint(0, H - crop_h + 1)
                left = np.random.randint(0, W - crop_w + 1)

                cropped = img[:, top:top + crop_h, left:left + crop_w]

                hwc = cropped.transpose(1, 2, 0)
                resized = cv2.resize(hwc, (self.size[1], self.size[0]),
                                     interpolation=interp_map[self.interpolation])
                return resized.transpose(2, 0, 1)

        # Fallback: center crop to the largest inscribed rectangle
        # that matches the target aspect ratio, then resize
        in_ratio = W / H
        target_ratio = self.size[1] / self.size[0]

        if in_ratio < target_ratio:
            # Image is too tall — crop height
            crop_w = W
            crop_h = int(round(W / target_ratio))
        else:
            # Image is too wide — crop width
            crop_h = H
            crop_w = int(round(H * target_ratio))

        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        cropped = img[:, top:top + crop_h, left:left + crop_w]

        hwc = cropped.transpose(1, 2, 0)
        resized = cv2.resize(hwc, (self.size[1], self.size[0]),
                             interpolation=interp_map[self.interpolation])
        return resized.transpose(2, 0, 1)



class FiveCrop:
    """Crop 5 regions: 4 corners + center."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError("TODO: Implement FiveCrop")


class TenCrop:
    """FiveCrop + horizontal flips = 10 crops."""

    def __init__(self, size: Union[int, Tuple[int, int]], vertical_flip: bool = False):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.vertical_flip = vertical_flip

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError("TODO: Implement TenCrop")


class Pad:
    """Pad image on all sides."""

    def __init__(
        self,
        padding: Union[int, Tuple[int, ...]],
        fill: Union[int, Tuple[int, ...]] = 0,
        padding_mode: str = 'constant',
    ):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement Pad")


class RandomPad:
    """Randomly pad image."""

    def __init__(self, padding: int, fill: int = 0, padding_mode: str = 'constant'):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomPad")


class RandomHorizontalFlip:
    """Randomly flip image horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.uniform() < self.p:
            img = cv2.flip(img, 1)
        return img


class RandomVerticalFlip:
    """Randomly flip image vertically."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.uniform() < self.p:
            img = cv2.flip(img, 0)
        return img


class RandomRotation:
    """Randomly rotate image."""

    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]],
        interpolation: str = 'nearest',
        expand: bool = False,
        center: Optional[Tuple[int, int]] = None,
        fill: Union[int, Tuple[int, ...]] = 0,
    ):
        self.degrees = degrees if isinstance(degrees, tuple) else (-degrees, degrees)
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomRotation")


class RandomAffine:
    """Random affine transformation."""

    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]],
        translate: Optional[Tuple[float, float]] = None,
        scale: Optional[Tuple[float, float]] = None,
        shear: Optional[Union[float, Tuple[float, ...]]] = None,
        interpolation: str = 'nearest',
        fill: Union[int, Tuple[int, ...]] = 0,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomAffine")


class RandomPerspective:
    """Random perspective transformation."""

    def __init__(
        self,
        distortion_scale: float = 0.5,
        p: float = 0.5,
        interpolation: str = 'bilinear',
        fill: Union[int, Tuple[int, ...]] = 0,
    ):
        self.distortion_scale = distortion_scale
        self.p = p
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomPerspective")


class ElasticTransform:
    """Elastic deformation of images."""

    def __init__(
        self,
        alpha: float = 50.0,
        sigma: float = 5.0,
        interpolation: str = 'bilinear',
        fill: Union[int, Tuple[int, ...]] = 0,
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement ElasticTransform")


class ScaleJitter:
    """Randomly scale image within range."""

    def __init__(
        self,
        target_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.1, 2.0),
        interpolation: str = 'bilinear',
    ):
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement ScaleJitter")


class RandomShortestSize:
    """Resize so shortest side is within range."""

    def __init__(
        self,
        min_size: Union[int, List[int]],
        max_size: int,
        interpolation: str = 'bilinear',
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomShortestSize")


class RandomZoomOut:
    """Randomly zoom out (place image on larger canvas)."""

    def __init__(
        self,
        fill: Union[int, Tuple[int, ...]] = 0,
        side_range: Tuple[float, float] = (1.0, 4.0),
        p: float = 0.5,
    ):
        self.fill = fill
        self.side_range = side_range
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Implement RandomZoomOut")
