# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
import math

import jax
import jax.numpy as jnp


def bias_linspace_initializer(start: float = 3.4, end: float = 6.0):
    """Returns an initializer function for linearly spaced biases."""

    def init_fn(key: jax.Array, shape: tuple[int, ...], dtype=jnp.float_):
        assert len(shape) == 1, (
            f"shape must be 1-dimensional (typically a bias), got {len(shape)}"
        )

        n_dims = shape[0]
        init_vals = jnp.linspace(start, end, n_dims, dtype=dtype)
        return init_vals

    return init_fn


def small_init_initializer(dim: int):
    """Returns an initializer function using the small init method."""

    def init_fn(key: jax.Array, shape: tuple[int, ...], dtype=jnp.float_):
        std = math.sqrt(2 / (5 * dim))
        return jax.random.normal(key, shape, dtype=dtype) * std

    return init_fn


def wang_initializer(dim: int, num_blocks: int):
    """Returns an initializer function using the Wang init method."""

    def init_fn(key: jax.Array, shape: tuple[int, ...], dtype=jnp.float_):
        std = 2 / num_blocks / math.sqrt(dim)
        return jax.random.normal(key, shape, dtype=dtype) * std

    return init_fn
