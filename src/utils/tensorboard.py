import logging
import typing as tp
from pathlib import Path

import jax
import jax.tree_util as jtu
import numpy as np
import optax
from flax.metrics.tensorboard import SummaryWriter


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
            image = np.array(image)

        # Ensure image is in the range [0, 1]
        if image.max() <= 1.0 and image.min() >= 0.0:
            pass  # Already in [0, 1]
        elif image.max() <= 255 and image.min() >= 0:
            image = image / 255.0  # Convert from [0, 255] to [0, 1]
        else:
            # Normalize to [0, 1]
            image = (image - image.min()) / (image.max() - image.min())

        self.writer.image(tag, image, step, max_outputs=max_outputs)

    def log_images(
        self,
        tag: str,
        images: tp.Union[jax.Array, np.ndarray],
        step: int,
        max_outputs: int = 8,
    ):
        """Log multiple images as a batch.

        Args:
            tag: Name for the images
            images: Images array with shape (B, H, W, C)
            step: Global step
            max_outputs: Maximum number of images to log
        """
        if isinstance(images, jax.Array):
            images = np.array(images)

        # Ensure we don't exceed max_outputs
        if images.shape[0] > max_outputs:
            images = images[:max_outputs]

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
