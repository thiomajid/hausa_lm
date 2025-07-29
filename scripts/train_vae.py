import json
import logging
import shutil
import time
import typing as tp
from dataclasses import asdict
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
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import HfArgumentParser

from src.data.image.data import create_dataloaders
from src.data.image.transforms import (
    LoadFromBytesAndResize,
    NormalizeImage,
    unnormalize_image,
)
from src.models.vae import VAE, vae_loss
from src.models.vae.vae import VAEConfig
from src.trainer.base_arguments import BaseTrainingArgs
from src.trainer.helpers import compute_training_steps
from src.utils.devices import create_mesh, log_node_devices_stats
from src.utils.modules import checkpoint_post_eval, count_parameters
from src.utils.tensorboard import TensorBoardLogger
from xlstm_jax.utils import str2dtype

logger = logging.getLogger(__name__)


def generate_epsilon(rngs: nnx.Rngs, batch_size: int, latent_dim: int) -> jax.Array:
    """Generate epsilon for reparameterization trick outside of jitted functions."""
    return jax.random.normal(rngs.sample(), (batch_size, latent_dim))


def generate_latent_codes(
    rngs: nnx.Rngs, num_samples: int, latent_dim: int
) -> jax.Array:
    """Generate latent codes from prior distribution outside of jitted functions."""
    return jax.random.normal(rngs.sample(), (num_samples, latent_dim))


def log_images_to_tensorboard(
    model: VAE,
    rngs: nnx.Rngs,
    tb_logger: TensorBoardLogger,
    step: int,
    mean: tuple[int, ...],
    std: tuple[int, ...],
    num_samples: int = 8,
    tag_prefix: str = "generated",
    real_images: jax.Array = None,
):
    """Log generated images to TensorBoard."""
    # Generate latent codes from prior distribution
    z = generate_latent_codes(rngs, num_samples, model.config.latent_dim)

    # Generate samples using the latent codes
    samples = model.generate(z)

    # Convert samples to numpy for logging
    samples_np = np.array(samples)

    # Unnormalize generated samples (assuming they are normalized like inputs)
    # For VAE outputs, we assume they match the input normalization
    samples_np = unnormalize_image(samples_np, mean, std)

    # If grayscale, add channel dimension
    if model.config.channels == 1 and samples_np.ndim == 3:
        samples_np = np.expand_dims(samples_np, axis=-1)

    # Log generated images to TensorBoard
    tb_logger.log_images(
        f"{tag_prefix}/samples", samples_np, step, max_outputs=num_samples
    )

    # Log real images for comparison if provided
    if real_images is not None:
        real_images_np = np.array(real_images)

        # Unnormalize real images (they are normalized with our transform)
        real_images_np = unnormalize_image(real_images_np, mean, std)

        # Take only the first num_samples for comparison
        if real_images_np.shape[0] > num_samples:
            real_images_np = real_images_np[:num_samples]

        # If grayscale, add channel dimension
        if model.config.channels == 1 and real_images_np.ndim == 3:
            real_images_np = np.expand_dims(real_images_np, axis=-1)

        tb_logger.log_images(
            f"{tag_prefix}/real", real_images_np, step, max_outputs=num_samples
        )

    logger.info(f"Logged {num_samples} generated images to TensorBoard at step {step}")


@partial(nnx.jit, static_argnames=("beta",))
def train_step(
    model: VAE,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch: jax.Array,
    eps: jax.Array,
    beta: float,
) -> jax.Array:
    """Single training step."""

    def loss_fn(model: VAE) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        reconstruction, mean, logvar = model(batch, eps=eps, training=True)
        total_loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction, batch, mean, logvar, beta
        )
        return total_loss, (total_loss, reconstruction_loss, kl_loss)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)

    total_loss, reconstruction_loss, kl_loss = aux

    # Update metrics
    metrics.update(
        loss=total_loss,
        reconstruction_loss=reconstruction_loss,
        kl_loss=kl_loss,
    )

    return total_loss


@partial(nnx.jit, static_argnames=("beta",))
def eval_step(
    model: VAE,
    metrics: nnx.MultiMetric,
    batch: jax.Array,
    beta: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single evaluation step."""
    reconstruction, mean, logvar = model(batch, training=False)
    total_loss, reconstruction_loss, kl_loss = vae_loss(
        reconstruction, batch, mean, logvar, beta
    )

    # Update metrics
    metrics.update(
        loss=total_loss,
        reconstruction_loss=reconstruction_loss,
        kl_loss=kl_loss,
    )

    return total_loss, reconstruction_loss, kl_loss


def save_generated_samples(
    model: VAE,
    rngs: nnx.Rngs,
    num_samples: int,
    output_path: str,
    mean: tuple[int, ...],
    std: tuple[int, ...],
):
    """Generate and save sample images."""
    # Generate latent codes from prior distribution
    z = generate_latent_codes(rngs, num_samples, model.config.latent_dim)

    # Generate samples using the latent codes
    samples = model.generate(z)

    # Convert samples to numpy and unnormalize
    samples_np = np.array(samples)
    samples_np = unnormalize_image(samples_np, mean, std)

    # Create a grid of images
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    # Handle single subplot case
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(num_samples):
        if model.config.channels == 1:
            # Grayscale image - squeeze the channel dimension for display
            if samples_np[i].ndim == 3:
                img = samples_np[i].squeeze(-1)
            else:
                img = samples_np[i]
            axes[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            # Color image
            axes[i].imshow(samples_np[i])
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

    logger.info("Starting VAE training...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config_dict = OmegaConf.to_container(cfg["model"], resolve=True)
    config = VAEConfig(**model_config_dict)

    mesh_shape = tuple(args.mesh_shape)
    axis_names = tuple(args.axes_names)
    mesh = create_mesh(mesh_shape=mesh_shape, axis_names=axis_names)

    dtype = str2dtype(cfg["dtype"])
    param_dtype = str2dtype(cfg["param_dtype"])
    rngs = nnx.Rngs(args.seed)

    model: VAE = None
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

    # Load dataset
    logger.info("Loading dataset...")
    IMAGE_COLUMN = cfg["image_column"]
    target_columns = [IMAGE_COLUMN]

    NORM_MEAN = tuple(cfg["norm_mean"])
    NORM_STD = tuple(cfg["norm_std"])

    # Create data transforms pipeline
    train_transforms = [
        LoadFromBytesAndResize(
            image_key=IMAGE_COLUMN,
            width=config.image_size,
            height=config.image_size,
        ),
        NormalizeImage(
            mean=NORM_MEAN,
            std=NORM_STD,
            image_key=IMAGE_COLUMN,
        ),
        grain.Batch(batch_size=args.per_device_train_batch_size, drop_remainder=True),
    ]

    train_loader, _ = create_dataloaders(
        logger=logger,
        args=args,
        target_columns=target_columns,
        train_transforms=train_transforms,
        eval_transforms=None,  # No evaluation needed for VAE training
    )

    # Setup the training loop
    num_train_samples = len(train_loader._data_source)

    logger.info(f"Dataset size - Train: {num_train_samples}")
    steps_dict = compute_training_steps(
        args,
        train_samples=num_train_samples,
        eval_samples=0,  # No evaluation for VAE training
        logger=logger,
    )

    max_steps = steps_dict["max_steps"]
    max_optimizer_steps = steps_dict["max_optimizer_steps"]

    # Set default warmup_ratio if not provided
    if not hasattr(args, "warmup_ratio"):
        args.warmup_ratio = 0.1
        logger.warning(
            f"warmup_ratio not found in config, defaulting to {args.warmup_ratio}"
        )

    # Use optimizer steps for learning rate schedule (not micro-batch steps)
    warmup_steps = int(args.warmup_ratio * max_optimizer_steps)
    logger.info(
        f"Calculated warmup steps: {warmup_steps} ({args.warmup_ratio=}, max_optimizer_steps={max_optimizer_steps})"
    )

    # Create warmup cosine learning rate schedule
    cosine_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=int(max_optimizer_steps - warmup_steps),
        end_value=args.learning_rate * 0.1,
    )

    logger.info(
        f"Using warmup cosine learning rate schedule: 0.0 -> {args.learning_rate} -> {args.learning_rate * 0.1} over {max_optimizer_steps} optimizer steps (warmup: {warmup_steps} steps)"
    )

    # Optimizer
    optimizer_def = optax.chain(
        optax.adam(
            learning_rate=args.learning_rate,
            b1=args.adam_beta1,
            b2=args.adam_beta2,
            # weight_decay=getattr(args, "weight_decay", 0.01),
        ),
    )

    optimizer_def = optax.MultiSteps(
        optimizer_def,
        every_k_schedule=args.gradient_accumulation_steps,
    )

    optimizer = nnx.Optimizer(model, optimizer_def)

    # Metrics
    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        reconstruction_loss=nnx.metrics.Average("reconstruction_loss"),
        kl_loss=nnx.metrics.Average("kl_loss"),
    )

    # TensorBoard logger
    tb_logger = TensorBoardLogger(log_dir=args.logging_dir, name="train")

    # Training loop setup
    global_step = 0
    global_optimizer_step = 0

    # Checkpoint manager
    CHECKPOINT_DIR = Path(cfg["checkpoint_save_dir"]).absolute()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    BEST_METRIC_KEY = "train_loss"
    CHECKPOINT_OPTIONS = ocp.CheckpointManagerOptions(
        max_to_keep=args.save_total_limit,
        best_fn=lambda metrics: metrics[BEST_METRIC_KEY],
        best_mode="min",
        create=True,
    )
    latest_eval_metrics_for_ckpt = {BEST_METRIC_KEY: float("inf")}

    # Manual best metric tracking
    best_metric_value = float("inf")
    best_step = None

    logger.info("Starting training loop...")
    logger.info(f"Num Epochs = {args.num_train_epochs}")
    logger.info(f"Micro Batch size = {args.per_device_train_batch_size}")
    logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"Effective Batch size = {args.per_device_train_batch_size * args.gradient_accumulation_steps}"
    )
    logger.info(f"Total batches per epoch: Train - {steps_dict['train_batches']}")
    logger.info(f"Total steps = {max_steps}")
    logger.info(f"Total optimizer steps = {max_optimizer_steps}")

    # Start timing
    training_start_time = time.perf_counter()
    epoch_durations = []

    # Training Loop
    DATA_SHARDING = NamedSharding(
        create_mesh(mesh_shape, axis_names),
        spec=P("dp", None),
    )

    beta = cfg.get("beta", 1.0)

    for epoch in range(args.num_train_epochs):
        epoch_start_time = time.perf_counter()
        logger.info(f"Starting Epoch {epoch + 1}/{args.num_train_epochs}")
        train_metrics.reset()

        epoch_desc = f"Epoch {epoch + 1}/{args.num_train_epochs}"
        with tqdm(
            total=steps_dict["steps_per_epoch"],
            desc=epoch_desc,
            leave=True,
        ) as pbar:
            pbar.set_description(epoch_desc)
            for step, batch in enumerate(train_loader):
                global_step += 1  # Count every batch as a step

                # Prepare batch - Convert to JAX arrays and get images
                batch_images = jnp.array(batch[IMAGE_COLUMN])

                # Handle extra batch dimensions (squeeze if ndim > 4)
                if batch_images.ndim > 4:
                    batch_images = batch_images.squeeze(0)

                # Apply data sharding
                batch_images = jax.device_put(batch_images, DATA_SHARDING)

                # Generate epsilon for reparameterization trick outside jitted function
                # We need epsilon with the same batch size and latent dimension
                batch_size = batch_images.shape[0]
                eps = generate_epsilon(rngs, batch_size, model.config.latent_dim)

                # Training step
                loss = train_step(
                    model=model,
                    optimizer=optimizer,
                    metrics=train_metrics,
                    batch=batch_images,
                    eps=eps,
                    beta=beta,
                )

                # Check if it's time for optimizer step
                is_update_step = (step + 1) % args.gradient_accumulation_steps == 0
                if is_update_step:
                    global_optimizer_step += 1

                    # Log learning rate
                    current_lr = cosine_schedule(global_optimizer_step)
                    tb_logger.log_learning_rate(current_lr, global_optimizer_step)

                # Logging
                if global_step % args.logging_steps == 0:
                    computed_metrics = train_metrics.compute()

                    # Log metrics to TensorBoard
                    for metric, value in computed_metrics.items():
                        tb_logger.log_scalar(f"train/{metric}", value, global_step)

                    train_metrics.reset()

                # Update progress bar
                current_lr = cosine_schedule(global_optimizer_step)
                postfix_data = {
                    "step": f"{global_step}/{max_steps}",
                    "opt_step": f"{global_optimizer_step}/{max_optimizer_steps}",
                    "lr": f"{current_lr:.2e}",
                    "loss": f"{loss.item():.6f}",
                }

                # Add best metrics
                if best_step is not None:
                    postfix_data["best_loss"] = f"{best_metric_value:.6f}"
                    postfix_data["best_step"] = str(best_step)

                current_desc = f"Epoch {epoch + 1}/{args.num_train_epochs} (Step {global_step}/{max_steps}, Opt {global_optimizer_step}/{max_optimizer_steps})"
                pbar.set_description(current_desc)
                pbar.set_postfix(postfix_data)
                pbar.update(1)

        # --- End of epoch processing ---
        epoch_end_time = time.perf_counter()

        # Get final training metrics for this epoch
        epoch_train_metrics = train_metrics.compute()
        current_train_loss = float(epoch_train_metrics.get("loss", float("inf")))

        # Log epoch training metrics to TensorBoard
        for metric, value in epoch_train_metrics.items():
            tb_logger.log_scalar(f"train_epoch/{metric}", value, global_step)

        # Update manual best metric tracking based on training loss
        if current_train_loss < best_metric_value:
            best_metric_value = current_train_loss
            best_step = global_step
            logger.info(
                f"New best {BEST_METRIC_KEY}: {best_metric_value:.6f} at step {best_step}"
            )

        # Update latest metrics for potential final checkpoint
        latest_eval_metrics_for_ckpt = {BEST_METRIC_KEY: current_train_loss}

        # Save checkpoint using train loss as metric
        checkpoint_post_eval(
            logger=logger,
            model=model,
            metrics=train_metrics,
            tb_logger=tb_logger,
            global_step=global_step,
            epoch=epoch,
            best_metric_key=BEST_METRIC_KEY,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_options=CHECKPOINT_OPTIONS,
        )

        # Generate and save sample images
        try:
            save_generated_samples(
                model,
                rngs,
                cfg.get("num_samples_to_generate", 16),
                output_dir / f"samples_epoch_{epoch + 1}.png",
                mean=NORM_MEAN,
                std=NORM_STD,
            )

            # Get a small batch from training data for comparison (real images and reconstructions)
            try:
                sample_batch = next(iter(train_loader))
                sample_images = jnp.array(sample_batch[IMAGE_COLUMN])

                # Handle extra batch dimensions (squeeze if ndim > 4)
                if sample_images.ndim > 4:
                    sample_images = sample_images.squeeze(0)

                sample_images = jax.device_put(sample_images, DATA_SHARDING)

                # Log reconstructions as well
                reconstructions, _, _ = model(sample_images[:8], training=False)

                # Log images to TensorBoard (generated, real, and reconstructions)
                log_images_to_tensorboard(
                    model,
                    rngs,
                    tb_logger,
                    global_step,
                    num_samples=8,
                    tag_prefix="epoch_end",
                    real_images=sample_images[:8],
                    mean=NORM_MEAN,
                    std=NORM_STD,
                )

                # Log reconstructions separately
                reconstructions_np = np.array(reconstructions)
                reconstructions_np = unnormalize_image(
                    reconstructions_np,
                    mean=NORM_MEAN,
                    std=NORM_STD,
                )

                if model.config.channels == 1 and reconstructions_np.ndim == 3:
                    reconstructions_np = np.expand_dims(reconstructions_np, axis=-1)

                tb_logger.log_images(
                    "epoch_end/reconstructions",
                    reconstructions_np,
                    global_step,
                    max_outputs=8,
                )

            except Exception as e:
                logger.warning(f"Could not get sample batch for comparison: {e}")
                # Fallback to just generated images
                log_images_to_tensorboard(
                    model,
                    rngs,
                    tb_logger,
                    global_step,
                    num_samples=8,
                    tag_prefix="epoch_end",
                    mean=NORM_MEAN,
                    std=NORM_STD,
                )

        except Exception as e:
            logger.warning(f"Could not generate sample images: {e}")

        # Record epoch duration and log to TensorBoard
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_durations.append(epoch_duration)
        tb_logger.log_scalar("timing/epoch_duration", epoch_duration, global_step)
        logger.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")

    logger.info("Training completed.")

    # Log final training metrics
    if train_metrics is not None:
        try:
            final_computed_metrics = train_metrics.compute()
            if final_computed_metrics and any(
                v.item() != 0 for v in final_computed_metrics.values()
            ):
                logger.info(
                    "Logging final training metrics from last accumulation cycle..."
                )

                key_metric = "loss"
                if key_metric in final_computed_metrics:
                    latest_eval_metrics_for_ckpt = {
                        BEST_METRIC_KEY: float(final_computed_metrics[key_metric])
                    }

                # Log final metrics to TensorBoard
                for metric, value in final_computed_metrics.items():
                    tb_logger.log_scalar(f"train/{metric}", value, global_step)

                current_lr = cosine_schedule(global_optimizer_step)
                tb_logger.log_learning_rate(current_lr, global_optimizer_step)
                logger.info(f"Final learning rate: {current_lr:.2e}")
        except Exception as e:
            logger.warning(f"Could not compute final training metrics: {e}")

    # Calculate total training duration and log to TensorBoard
    training_end_time = time.perf_counter()
    total_training_duration = training_end_time - training_start_time
    tb_logger.log_scalar(
        "timing/total_training_duration", total_training_duration, global_step
    )

    # Calculate and log timing statistics
    avg_epoch_duration = (
        sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0
    )

    tb_logger.log_scalar("timing/avg_epoch_duration", avg_epoch_duration, global_step)

    logger.info(
        f"Training completed in {total_training_duration:.2f} seconds ({total_training_duration / 3600:.2f} hours)"
    )
    logger.info(f"Average epoch duration: {avg_epoch_duration:.2f} seconds")

    # Log best metric summary
    if best_step is not None:
        logger.info(
            f"Best {BEST_METRIC_KEY}: {best_metric_value:.6f} achieved at step {best_step}"
        )
    else:
        logger.warning("No best checkpoint was identified during training.")

    # Close TensorBoard logger
    tb_logger.close()

    # Final saving and upload
    logger.info("Saving final artifacts...")
    artifacts_dir = Path(args.output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Copy TensorBoard logs to artifacts directory
    tb_logs_source = Path(args.logging_dir)
    tb_logs_target = artifacts_dir / "tensorboard_logs"
    if tb_logs_source.exists():
        shutil.copytree(tb_logs_source, tb_logs_target, dirs_exist_ok=True)
        logger.info(f"TensorBoard logs copied to {tb_logs_target}")

    # Save training history
    training_summary = {
        "total_training_duration": total_training_duration,
        "avg_epoch_duration": avg_epoch_duration,
        "num_epochs_completed": len(epoch_durations),
        "global_steps": global_step,
        "global_optimizer_steps": global_optimizer_step,
    }
    with open(artifacts_dir / "train_history.json", "w") as f:
        json.dump(training_summary, f, indent=4)
    logger.info(f"Training history saved to {artifacts_dir / 'train_history.json'}")

    # Save model config
    with open(artifacts_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=4)
    logger.info(f"Model config saved to {artifacts_dir / 'config.json'}")

    # Save trainer config
    with open(artifacts_dir / "trainer_config.json", "w") as f:
        trainer_config_dict = asdict(args)
        if "hub_token" in trainer_config_dict:
            trainer_config_dict.pop("hub_token")
        json.dump(trainer_config_dict, f, indent=4)
    logger.info(f"Trainer config saved to {artifacts_dir / 'trainer_config.json'}")

    # Save timing summary
    timing_summary = {
        "total_training_duration_seconds": total_training_duration,
        "total_training_duration_hours": total_training_duration / 3600,
        "average_epoch_duration_seconds": avg_epoch_duration,
        "num_epochs_completed": len(epoch_durations),
        "num_checkpoints_saved": len(epoch_durations),  # One checkpoint per epoch
    }
    with open(artifacts_dir / "timing_summary.json", "w") as f:
        json.dump(timing_summary, f, indent=4)
    logger.info(f"Timing summary saved to {artifacts_dir / 'timing_summary.json'}")

    # Save final model state
    if global_step > 0:
        final_model_state = nnx.state(model, nnx.Param)
        logger.info(
            f"Saving final model state at step {global_step} with metrics {latest_eval_metrics_for_ckpt}."
        )

        # Save final checkpoint
        with ocp.CheckpointManager(
            CHECKPOINT_DIR,
            options=CHECKPOINT_OPTIONS,
            checkpointers=ocp.PyTreeCheckpointer(),
        ) as manager:
            manager.save(
                global_step,
                args=ocp.args.PyTreeSave(final_model_state),
                metrics=latest_eval_metrics_for_ckpt,
            )

    # Copy best checkpoint to artifacts directory
    best_step_to_deploy = best_step
    target_ckpt_deployment_path = artifacts_dir / "model_checkpoint"

    if best_step_to_deploy is not None:
        logger.info(
            f"Best checkpoint according to manual tracking is at step {best_step_to_deploy} (based on {BEST_METRIC_KEY}: {best_metric_value:.6f})."
        )
        source_ckpt_dir = CHECKPOINT_DIR / str(best_step_to_deploy)

        if source_ckpt_dir.exists():
            logger.info(
                f"Copying best checkpoint from {source_ckpt_dir} to {target_ckpt_deployment_path}"
            )
            if target_ckpt_deployment_path.exists():
                shutil.rmtree(target_ckpt_deployment_path)

            target_ckpt_deployment_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                source_ckpt_dir, target_ckpt_deployment_path, dirs_exist_ok=True
            )
        else:
            logger.error(f"Best checkpoint directory {source_ckpt_dir} not found.")
    else:
        logger.warning("No best checkpoint identified through manual tracking.")
        if global_step > 0:
            final_model_state = nnx.state(model, nnx.Param)
            logger.info(
                f"Saving current final model state directly to {target_ckpt_deployment_path} as a fallback."
            )
            if target_ckpt_deployment_path.exists():
                shutil.rmtree(target_ckpt_deployment_path)
            target_ckpt_deployment_path.mkdir(parents=True, exist_ok=True)

            # Use PyTreeCheckpointer for fallback save
            ocp.PyTreeCheckpointHandler().save(
                target_ckpt_deployment_path,
                final_model_state,
            )
        else:
            logger.error(
                "No optimizer steps were completed, and no best checkpoint found. Cannot save a model."
            )

    # Save final samples
    try:
        save_generated_samples(
            model,
            rngs,
            cfg.get("num_samples_to_generate", 16),
            artifacts_dir / "final_samples.png",
            mean=NORM_MEAN,
            std=NORM_STD,
        )
        logger.info("Final sample images saved.")
    except Exception as e:
        logger.warning(f"Could not save final sample images: {e}")

    logger.info("Training and artifact saving completed!")


@hydra.main(version_base="1.2", config_path="../configs", config_name="train_vae")
def main(cfg: DictConfig) -> None:
    """Main entry point."""

    train_vae(cfg)


if __name__ == "__main__":
    main()
