import logging
import typing as tp
from pathlib import Path

import jax
import jax.tree_util as jtu
import numpy as np
import optax
from flax.metrics.tensorboard import SummaryWriter


def make_grid(
    images: np.ndarray,
    nrow: tp.Optional[int] = None,
    padding: int = 2,
    normalize: bool = False,
    value_range: tp.Optional[tp.Tuple[float, float]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Make a grid of images similar to torchvision.utils.make_grid.

    Args:
        images: Input images with shape (B, H, W, C) or (B, H, W)
        nrow: Number of images displayed in each row of the grid.
              If None, it's set to the square root of the number of images.
        padding: Amount of padding between images
        normalize: If True, shift the image to the range (0, 1)
        value_range: Tuple (min, max) where min and max are numbers,
                    then these numbers are used to normalize the image.
        scale_each: If True, scale each image in the batch of images separately
        pad_value: Value for the padded pixels

    Returns:
        Grid image with shape (H, W, C) or (H, W)
    """
    if not isinstance(images, np.ndarray):
        images = np.array(images)

    if images.ndim < 3 or images.ndim > 4:
        raise ValueError(f"Images should be 3D or 4D array, got {images.ndim}D")

    # Handle grayscale images (B, H, W) -> (B, H, W, 1)
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)

    nmaps, height, width, channels = images.shape

    if nrow is None:
        nrow = int(np.ceil(np.sqrt(nmaps)))

    # Calculate grid dimensions
    nrows = int(np.ceil(nmaps / nrow))

    # Create the grid
    grid_height = nrows * height + (nrows - 1) * padding
    grid_width = nrow * width + (nrow - 1) * padding

    if channels == 1:
        grid = np.full((grid_height, grid_width), pad_value, dtype=images.dtype)
    else:
        grid = np.full(
            (grid_height, grid_width, channels), pad_value, dtype=images.dtype
        )

    # Normalize if requested
    if normalize or value_range is not None or scale_each:
        images = images.copy()

        if scale_each:
            for i in range(nmaps):
                img = images[i]
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    images[i] = (img - img_min) / (img_max - img_min)
        elif value_range is not None:
            min_val, max_val = value_range
            images = np.clip(images, min_val, max_val)
            images = (images - min_val) / (max_val - min_val)
        elif normalize:
            img_min, img_max = images.min(), images.max()
            if img_max > img_min:
                images = (images - img_min) / (img_max - img_min)

    # Fill the grid
    for i in range(nmaps):
        row = i // nrow
        col = i % nrow

        y_start = row * (height + padding)
        y_end = y_start + height
        x_start = col * (width + padding)
        x_end = x_start + width

        if channels == 1:
            grid[y_start:y_end, x_start:x_end] = images[i, :, :, 0]
        else:
            grid[y_start:y_end, x_start:x_end] = images[i]

    # Remove single channel dimension if present
    if channels == 1:
        return grid
    else:
        return grid


class TensorBoardLogger:
    """Utility class for logging metrics and images to TensorBoard."""

    def __init__(self, log_dir: tp.Union[str, Path], name: str = "train"):
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
            name: Name for this logger instance
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.name = name

        # Create tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir / name))

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TensorBoard logging initialized at {self.log_dir / name}")

    def log_scalar(
        self,
        tag: str,
        value: tp.Union[float, jax.Array, np.ndarray],
        step: int,
    ):
        """Log a scalar value.

        Args:
            tag: Name of the scalar
            value: Scalar value to log
            step: Global step
        """
        if isinstance(value, (jax.Array, np.ndarray)):
            value = float(value.item())
        elif not isinstance(value, (int, float)):
            value = float(value)

        self.writer.scalar(tag, value, step)

    def log_scalars(
        self,
        tag_scalar_dict: tp.Dict[str, tp.Union[float, jax.Array, np.ndarray]],
        step: int,
    ):
        """Log multiple scalars at once.

        Args:
            tag_scalar_dict: Dictionary of tag -> scalar value
            step: Global step
        """
        for tag, value in tag_scalar_dict.items():
            self.log_scalar(tag, value, step)

    def log_histogram(
        self,
        tag: str,
        values: tp.Union[jax.Array, np.ndarray],
        step: int,
    ):
        """Log a histogram of values.

        Args:
            tag: Name for the histogram
            values: Values to create histogram from
            step: Global step
        """
        if isinstance(values, jax.Array):
            values = np.array(values)

        self.writer.histogram(tag, values, step)

    def log_learning_rate(self, lr: float, step: int):
        """Log learning rate.

        Args:
            lr: Learning rate value
            step: Global step
        """
        self.log_scalar("learning_rate", lr, step)

    def log_image(
        self,
        tag: str,
        image: tp.Union[jax.Array, np.ndarray],
        step: int,
        max_outputs: int = 3,
    ):
        """Log a single image or batch of images.

        Args:
            tag: Name for the image
            image: Image array with shape (H, W, C) or (B, H, W, C)
            step: Global step
            max_outputs: Maximum number of images to log if batch
        """
        if isinstance(image, jax.Array):
            image = np.array(jax.device_get(image))

        # Ensure image is in the range [0, 1]
        if image.max() <= 1.0 and image.min() >= 0.0:
            pass  # Already in [0, 1]
        elif image.max() <= 255 and image.min() >= 0:
            image = image / 255.0  # Convert from [0, 255] to [0, 1]
        else:
            # Normalize to [0, 1]
            image = (image - image.min()) / (image.max() - image.min())

        self.writer.image(tag, image, step, max_outputs=max_outputs)
        # if image.ndim != 4:
        # else:
        #     N = image.shape[0]
        #     for idx in range(N):
        #         self.writer.image(
        #             f"tag_{idx}", image[idx], step, max_outputs=max_outputs
        #         )

    def log_images(
        self,
        tag: str,
        images: tp.Union[jax.Array, np.ndarray],
        step: int,
        max_outputs: int = 8,
        nrow: tp.Optional[int] = None,
        as_grid: bool = True,
    ):
        """Log multiple images as a batch or grid.

        Args:
            tag: Name for the images
            images: Images array with shape (B, H, W, C) or (B, H, W)
            step: Global step
            max_outputs: Maximum number of images to log
            nrow: Number of images per row in grid (if as_grid=True)
            as_grid: If True, create a grid of images; if False, log individual images
        """
        if isinstance(images, jax.Array):
            images = np.array(jax.device_get(images))

        # Ensure we don't exceed max_outputs
        if images.shape[0] > max_outputs:
            images = images[:max_outputs]

        if as_grid and images.shape[0] > 1:
            # Create a grid of images
            grid_image = make_grid(images, nrow=nrow, padding=2, normalize=True)
            # Add batch dimension and channel dimension if needed
            if grid_image.ndim == 2:  # Grayscale
                grid_image = np.expand_dims(grid_image, axis=(0, -1))  # (1, H, W, 1)
            elif grid_image.ndim == 3:  # Color
                grid_image = np.expand_dims(grid_image, axis=0)  # (1, H, W, C)

            self.log_image(tag, grid_image, step, max_outputs=1)
        else:
            # Log individual images
            self.log_image(tag, images, step, max_outputs=max_outputs)

    def log_gradients(self, grads: tp.Dict[str, tp.Any], step: int):
        """Log gradient statistics.

        Args:
            grads: Dictionary of gradients
            step: Global step
        """

        # Compute gradient norm
        grad_norm = optax.global_norm(grads)
        self.log_scalar("gradient_norm", grad_norm, step)

        # Log histogram of gradient values for key layers
        def log_grad_hist(path, grad):
            if isinstance(grad, jax.Array) and grad.size > 0:
                tag = f"gradients/{'/'.join(map(str, path))}"
                self.log_histogram(tag, grad, step)

        # Traverse gradient tree and log histograms for some key components
        jtu.tree_map_with_path(log_grad_hist, grads)

    def close(self):
        """Close the TensorBoard writer."""
        if hasattr(self, "writer"):
            self.writer.close()
            self.logger.info(f"TensorBoard logger closed for {self.name}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
