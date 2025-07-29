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
    """Transform for normalizing image data."""

    def __init__(self, image_key: str = "image"):
        super().__init__()
        self.image_key = image_key

    def map(self, element):
        """Normalize a single image."""
        if self.image_key not in element:
            return element

        img = element[self.image_key]

        # Keep as numpy array - do NOT convert to JAX array
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Normalize to [0, 1] using numpy operations only
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)

        element[self.image_key] = img
        return element
