"""
Training script for Variational Autoencoder using Flax NNX and Hydra configuration.
"""

import logging
import time
import typing as tp
from functools import partial
from pathlib import Path
from pprint import pprint

import grain.python as grain
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
from flax import nnx
from jax.sharding import Mesh
from omegaconf import DictConfig, OmegaConf
from transformers import HfArgumentParser

from src.data.image.data import create_dataloaders
from src.data.image.transforms import LoadFromBytesAndResize, NormalizeImage
from src.models.vae import VAE, vae_loss
from src.models.vae.vae import VAEConfig
from src.trainer.base_arguments import BaseTrainingArgs
from src.trainer.helpers import compute_training_steps
from src.utils.devices import create_mesh, log_node_devices_stats
from src.utils.modules import count_parameters
from xlstm_jax.utils import str2dtype

logger = logging.getLogger(__name__)


def log_images_to_tensorboard(
    model: VAE,
    rngs: nnx.Rngs,
    writer,
    step: int,
    num_samples: int = 8,
    tag_prefix: str = "generated",
):
    """Log generated images to TensorBoard."""
    # Generate samples
    samples = model.generate(num_samples, rngs=rngs)

    # Convert to format expected by TensorBoard (NHWC)
    if model.config.channels == 1:
        # Add channel dimension for grayscale
        samples = jnp.expand_dims(samples, axis=-1)

    # Log to TensorBoard
    with writer.as_default():
        tf.summary.image(
            f"{tag_prefix}/samples", samples, step=step, max_outputs=num_samples
        )


@partial(nnx.jit, static_argnames=("beta",))
def train_step(
    model: VAE,
    optimizer: nnx.Optimizer,
    batch: jax.Array,
    beta: float,
    rngs: nnx.Rngs,
) -> tuple[dict[str, float], nnx.Rngs]:
    """Single training step."""

    def loss_fn(model: VAE) -> tuple[jax.Array, dict[str, float]]:
        reconstruction, mean, logvar = model(batch, rngs=rngs, training=True)
        loss, loss_dict = vae_loss(reconstruction, batch, mean, logvar, beta)
        return loss, loss_dict

    (loss, loss_dict), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)

    return loss_dict, rngs


@partial(nnx.jit, static_argnames=("beta",))
def eval_step(
    model: VAE,
    batch: jax.Array,
    rngs: nnx.Rngs,
    beta: float,
) -> dict[str, float]:
    """Single evaluation step."""
    reconstruction, mean, logvar = model(batch, rngs=rngs, training=False)
    loss, loss_dict = vae_loss(reconstruction, batch, mean, logvar, beta)
    return loss_dict


def save_generated_samples(
    model: VAE,
    rngs: nnx.Rngs,
    num_samples: int,
    output_path: str,
):
    """Generate and save sample images."""
    # Generate samples in proper image format
    samples = model.generate(num_samples, rngs=rngs)

    # Create a grid of images
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(num_samples):
        if model.config.channels == 1:
            # Grayscale image
            axes[i].imshow(samples[i], cmap="gray", vmin=0, vmax=1)
        else:
            # Color image
            axes[i].imshow(samples[i])
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


@partial(
    nnx.jit,
    static_argnames=("mesh", "dtype", "config_fn", "param_dtype"),
)
def _create_sharded_model(
    rngs: nnx.Rngs,
    config_fn: tp.Callable[[], VAEConfig],
    mesh: Mesh,
    dtype=jnp.bfloat16,
    param_dtype=jnp.float32,
):
    """Create and shard a HuggingFace model."""
    config = config_fn()
    model = VAE(config, mesh=mesh, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)

    return model


def train_vae(cfg: DictConfig):
    """Main training function."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = HfArgumentParser(BaseTrainingArgs)

    # Load trainer arguments from YAML file
    args = parser.parse_dict(OmegaConf.to_container(cfg["trainer"], resolve=True))[0]
    args = tp.cast(BaseTrainingArgs, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config_dict = OmegaConf.to_container(cfg["model"], resolve=True)
    config = VAEConfig(**model_config_dict)
    model: VAE = None

    mesh_shape = tuple(args.mesh_shape)
    axis_names = tuple(args.axis_names)
    mesh = create_mesh(mesh_shape=mesh_shape, axis_names=axis_names)

    dtype = str2dtype(cfg["dtype"])
    param_dtype = str2dtype(cfg["param_dtype"])
    rngs = nnx.Rngs(args.seed)

    with mesh:
        model = _create_sharded_model(
            rngs,
            config_fn=lambda: config,
            mesh=mesh,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    logger.info("Model parameters")
    pprint(count_parameters(model))
    log_node_devices_stats(logger)

    # Checkpoint manager
    CHECKPOINT_DIR = Path(cfg["checkpoint_save_dir"]).absolute()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    BEST_METRIC_KEY = "eval_loss"
    CHECKPOINT_OPTIONS = ocp.CheckpointManagerOptions(
        max_to_keep=args.save_total_limit,
        best_fn=lambda metrics: metrics[BEST_METRIC_KEY],
        best_mode="min",
        create=True,
    )

    # Setup random number generators

    # Load dataset
    logger.info("Loading dataset...")
    image_column = cfg["image_column"]
    target_columns = [image_column]

    # Create data transforms pipeline
    train_transforms = [
        LoadFromBytesAndResize(
            image_key=image_column,
            width=config.image_width,
            height=config.image_height,
        ),
        NormalizeImage(image_key=image_column),
        grain.Batch(batch_size=args.per_device_train_batch_size, drop_remainder=True),
    ]

    eval_transforms = [
        LoadFromBytesAndResize(
            image_key=image_column,
            width=config.image_width,
            height=config.image_height,
        ),
        NormalizeImage(image_key=image_column),
        grain.Batch(batch_size=args.per_device_eval_batch_size, drop_remainder=True),
    ]

    train_loader, eval_loader = create_dataloaders(
        logger=logger,
        args=args,
        target_columns=target_columns,
        train_transforms=train_transforms,
        eval_transforms=eval_transforms,
    )

    # Setup the training loop
    num_train_samples = len(train_loader._data_source)
    num_eval_samples = len(eval_loader._data_source)

    logger.info(f"Dataset sizes - Train: {num_train_samples}, Eval: {num_eval_samples}")
    steps_dict = compute_training_steps(
        args,
        train_samples=num_train_samples,
        eval_samples=num_eval_samples,
        logger=logger,
    )

    max_steps = steps_dict["max_steps"]
    max_optimizer_steps = steps_dict["max_optimizer_steps"]

    # # Set default warmup_ratio if not provided
    # if not hasattr(args, "warmup_ratio"):
    #     args.warmup_ratio = 0.2
    #     logger.warning(
    #         f"warmup_ratio not found in config, defaulting to {args.warmup_ratio}"
    #     )

    # # Use optimizer steps for learning rate schedule (not micro-batch steps)
    # warmup_steps = int(args.warmup_ratio * max_optimizer_steps)
    # logger.info(
    #     f"Calculated warmup steps: {warmup_steps} ({args.warmup_ratio=}, max_optimizer_steps={max_optimizer_steps})"
    # )

    # # Create warmup cosine learning rate schedule
    # cosine_schedule = optax.warmup_cosine_decay_schedule(
    #     init_value=0.0,
    #     peak_value=args.learning_rate,
    #     warmup_steps=warmup_steps,
    #     decay_steps=int(max_optimizer_steps - warmup_steps),
    #     end_value=args.learning_rate * 0.2,
    # )

    # logger.info(
    #     f"Using warmup cosine learning rate schedule: 0.0 -> {args.learning_rate} -> {args.learning_rate * 0.2} over {max_optimizer_steps} optimizer steps (warmup: {warmup_steps} steps)"
    # )

    # Optimizer
    optimizer_def = optax.chain(
        optax.adam(
            learning_rate=args.learning_rate,
            b1=args.adam_beta1,
            b2=args.adam_beta2,
        ),
    )

    optimizer_def = optax.MultiSteps(
        optimizer_def,
        every_k_schedule=args.gradient_accumulation_steps,
    )

    optimizer = nnx.Optimizer(model, optimizer_def)

    # Setup TensorBoard
    log_dir = output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(str(log_dir))

    # Training loop
    logger.info("Starting training...")
    step = 0

    for epoch in range(args.num_train_epochs):
        model.train()

        epoch_start_time = time.time()
        epoch_losses = []

        # Training
        for batch in train_loader:
            # Convert to JAX arrays and get images
            batch_images = jnp.array(batch["image"])

            # Training step
            loss_dict, rngs = train_step(
                model,
                optimizer,
                batch_images,
                cfg.beta,
                rngs,
            )

            epoch_losses.append(loss_dict)

            # Logging
            if step % args.log_every == 0:
                avg_loss = jnp.mean(
                    jnp.array(
                        [
                            loss_dict["total_loss"]
                            for loss_dict in epoch_losses[-cfg.logging.log_every :]
                        ]
                    )
                )
                logger.info(f"Step {step}, Loss: {avg_loss:.4f}")

                # Log to TensorBoard
                with writer.as_default():
                    tf.summary.scalar("train/total_loss", avg_loss, step=step)
                    tf.summary.scalar(
                        "train/reconstruction_loss",
                        loss_dict["reconstruction_loss"],
                        step=step,
                    )
                    tf.summary.scalar("train/kl_loss", loss_dict["kl_loss"], step=step)

            # Evaluation
            if step % cfg.logging.eval_every == 0:
                eval_losses = []
                for eval_batch in eval_loader:
                    eval_batch_images = jnp.array(eval_batch["image"])
                    eval_loss_dict = eval_step(
                        model, {"images": eval_batch_images}, cfg.training.beta, rngs
                    )
                    eval_losses.append(eval_loss_dict)

                    # Only evaluate on a few batches to save time
                    if len(eval_losses) >= 10:
                        break

                avg_eval_loss = jnp.mean(
                    jnp.array([loss_dict["total_loss"] for loss_dict in eval_losses])
                )
                logger.info(f"Step {step}, Eval Loss: {avg_eval_loss:.4f}")

                # Log to TensorBoard
                with writer.as_default():
                    tf.summary.scalar("eval/total_loss", avg_eval_loss, step=step)

            # Generate samples
            if step % cfg.logging.checkpoint_every == 0:
                save_generated_samples(
                    model,
                    rngs,
                    cfg.logging.num_samples_to_generate,
                    output_dir / f"samples_step_{step}.png",
                )

            step += 1

        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = jnp.mean(
            jnp.array([loss_dict["total_loss"] for loss_dict in epoch_losses])
        )
        logger.info(
            f"Epoch {epoch + 1}/{cfg.training.num_epochs}, "
            f"Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s"
        )

        # Log generated images to TensorBoard at end of each epoch
        log_images_to_tensorboard(
            model,
            rngs,
            writer,
            epoch,
            num_samples=cfg.logging.get("tensorboard_samples", 8),
            tag_prefix="epoch_end",
        )

        # Log epoch metrics to TensorBoard
        with writer.as_default():
            tf.summary.scalar("epoch/loss", avg_epoch_loss, step=epoch)
            tf.summary.scalar("epoch/time", epoch_time, step=epoch)

    logger.info("Training completed!")

    # Save final samples
    save_generated_samples(
        model,
        rngs,
        cfg.logging.num_samples_to_generate,
        output_dir / "final_samples.png",
    )


@hydra.main(version_base=None, config_path="../configs", config_name="train_vae")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    logger.info("Starting VAE training with config:")
    logger.info(OmegaConf.to_yaml(cfg))

    train_vae(cfg)


if __name__ == "__main__":
    main()
