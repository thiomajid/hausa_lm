import logging
import math
import typing as tp

from flax import nnx

from src.trainer.base_arguments import BaseTrainingArgs


class TrainingSteps(tp.TypedDict):
    train_batches: int
    eval_batches: int
    max_steps: int
    max_optimizer_steps: int
    steps_per_epoch: int
    opt_steps_per_epoch: int


def compute_training_steps(
    args: BaseTrainingArgs,
    train_samples: int,
    eval_samples: int,
    logger: logging.Logger,
) -> TrainingSteps:
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    # Calculate number of batches per epoch
    train_batches = train_samples // args.per_device_train_batch_size
    eval_batches = eval_samples // args.per_device_eval_batch_size

    logger.info(f"Batches per epoch - Train: {train_batches}, Eval: {eval_batches}")

    # Each batch is processed as one step
    steps_per_epoch = train_batches

    # Optimizer updates happen every gradient_accumulation_steps batches
    optimizer_steps_per_epoch = math.ceil(
        train_batches // args.gradient_accumulation_steps
    )

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}")

    if optimizer_steps_per_epoch == 0:
        logger.warning(
            f"Number of batches per epoch ({train_batches}) is less than gradient_accumulation_steps ({args.gradient_accumulation_steps}). "
            "Effective optimizer steps per epoch is 0. Consider reducing accumulation steps or increasing dataset size."
        )

    max_steps = int(args.num_train_epochs * steps_per_epoch)
    max_optimizer_steps = int(args.num_train_epochs * optimizer_steps_per_epoch)

    return {
        "train_batches": train_batches,
        "eval_batches": eval_batches,
        "max_steps": max_steps,
        "max_optimizer_steps": max_optimizer_steps,
        "steps_per_epoch": steps_per_epoch,
        "opt_steps_per_epoch": optimizer_steps_per_epoch,
    }


@nnx.jit
def apply_gradients(optimizer: nnx.Optimizer, grads: nnx.State):
    """Apply accumulated gradients to the model using the optimizer."""
    optimizer.update(grads)
