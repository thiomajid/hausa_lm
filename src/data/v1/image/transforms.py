import io

import chex
import grain.python as grain
import numpy as np
from PIL import Image
from PIL.Image import Resampling


class LoadFromBytesAndResize(grain.MapTransform):
    """Transform to load images from bytes data and convert to numpy arrays."""

    def __init__(
        self,
        height: int,
        width: int,
        image_key: str = "image",
        is_rgb: bool = True,
    ):
        super().__init__()
        self.image_key = image_key
        self.is_rgb = is_rgb
        self.width = width
        self.height = height
        self.in_channels = 3 if is_rgb else 1

    def map(self, element):
        """Load a single image from bytes."""
        if self.image_key not in element:
            return element

        image_bytes = None
        image_data = element[self.image_key]

        # Handle different image data formats
        if isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]
        elif isinstance(image_data, bytes):
            image_bytes = image_data

        # Convert bytes to PIL Image and then to numpy
        image = None
        if isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        else:
            image = Image.open(io.BytesIO(image_bytes))

        image = image.convert("RGB") if self.is_rgb else image
        image = image.resize(
            size=(self.width, self.height), resample=Resampling.BILINEAR
        )
        image_array = np.array(image)

        chex.assert_shape(image_array, (self.height, self.width, self.in_channels))

        element[self.image_key] = image_array
        return element


class NormalizeImage(grain.MapTransform):
    """Transform for normalizing image data using PyTorch-style normalization with configurable mean and std.

    Formula: ((pixel / 255.0) - mean) / std
    First converts pixel values from [0, 255] to [0, 1], then applies mean/std normalization.
    This results in pixel values typically in [-1, 1] range when using standard ImageNet normalization.
    """

    def __init__(
        self,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        image_key: str = "image",
    ):
        super().__init__()
        self.image_key = image_key
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def map(self, element):
        """Normalize a single image using PyTorch-style normalization."""
        if self.image_key not in element:
            return element

        img = element[self.image_key]

        if not isinstance(img, np.ndarray):
            img = np.array(img)

        img = img.astype(np.float32)

        # First convert from [0, 255] to [0, 1] range
        img = img / 255.0

        # Apply PyTorch-style normalization: (pixel - mean) / std
        if img.ndim == 3 and img.shape[-1] == len(self.mean):
            img = (img - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            # Grayscale image with channel dimension
            # Use the first channel's mean and std for grayscale
            img = (img - self.mean[0]) / self.std[0]
        elif img.ndim == 2:
            # Grayscale image without channel dimension
            img = (img - self.mean[0]) / self.std[0]
        else:
            raise ValueError(
                f"Unsupported image shape: {img.shape}. Expected (H, W), (H, W, 1), or (H, W, 3)"
            )

        element[self.image_key] = img
        return element


def unnormalize_image(
    img: np.ndarray,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> np.ndarray:
    """Unnormalize an image or batch of images that was normalized with PyTorch-style normalization.

    Formula: pixel_unnormalized = (pixel_normalized * std) + mean

    Args:
        img: Normalized image array. Supports:
            - Single images: (H, W), (H, W, 1), or (H, W, C)
            - Batch of images: (B, H, W), (B, H, W, 1), or (B, H, W, C)
        mean: Mean values used for normalization
        std: Std values used for normalization

    Returns:
        Unnormalized image(s) in [0, 1] range
    """
    _mean = np.array(mean, dtype=np.float32)
    _std = np.array(std, dtype=np.float32)

    # Apply inverse normalization: pixel_unnormalized = (pixel_normalized * std) + mean
    if img.ndim == 4 and img.shape[-1] == len(mean):
        # Batch of multi-channel images (B, H, W, C)
        unnormalized = (img * _std.reshape(1, 1, 1, -1)) + _mean.reshape(1, 1, 1, -1)
    elif img.ndim == 4 and img.shape[-1] == 1:
        # Batch of grayscale images with channel dimension (B, H, W, 1)
        unnormalized = (img * _std[0]) + _mean[0]
    elif img.ndim == 3 and img.shape[-1] == len(mean):
        # Single multi-channel image (H, W, C)
        unnormalized = (img * _std.reshape(1, 1, -1)) + _mean.reshape(1, 1, -1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        # Single grayscale image with channel dimension (H, W, 1)
        unnormalized = (img * _std[0]) + _mean[0]
    elif img.ndim == 3 and len(mean) == 1:
        # Batch of grayscale images without channel dimension (B, H, W)
        unnormalized = (img * _std[0]) + _mean[0]
    elif img.ndim == 2:
        # Single grayscale image without channel dimension (H, W)
        unnormalized = (img * _std[0]) + _mean[0]
    else:
        raise ValueError(
            f"Unsupported image shape: {img.shape}. Expected shapes: (B,H,W,C), (B,H,W,1), (B,H,W), (H,W,C), (H,W,1), or (H,W)"
        )

    # Clip to [0, 1] range
    return np.clip(unnormalized, 0.0, 1.0)
