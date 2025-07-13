# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from ..utils import UpProjConfigMixin
from .init import small_init_initializer, wang_initializer

# Define activation functions using JAX
_act_fn_registry = {
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "relu^2": lambda x: jnp.square(jax.nn.relu(x)),
    "sigmoid": jax.nn.sigmoid,
    "swish": jax.nn.swish,
    "selu": jax.nn.selu,
}


def get_act_fn(act_fn_name: str) -> Callable[[jax.Array], jax.Array]:
    """Get activation function by name."""
    if act_fn_name in _act_fn_registry:
        return _act_fn_registry[act_fn_name]
    else:
        raise ValueError(
            f'Unknown activation function name "{act_fn_name}". '
            f"Available activation functions are: {list(_act_fn_registry.keys())}"
        )


@dataclass(unsafe_hash=True, order=True)
class FeedForwardConfig(UpProjConfigMixin):
    proj_factor: float = 1.3
    act_fn: str = "gelu"
    embedding_dim: int = -1
    dropout: float = 0.0
    bias: bool = False
    ff_type: Literal["ffn_gated"] = "ffn_gated"

    _num_blocks: int = 1

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        assert self.act_fn in _act_fn_registry, (
            f"Unknown activation function {self.act_fn}"
        )


class GatedFeedForward(nnx.Module):
    config_class = FeedForwardConfig

    def __init__(
        self,
        config: FeedForwardConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        # Initialize linear layers
        self.proj_up = nnx.Linear(
            in_features=config.embedding_dim,
            out_features=2 * config._proj_up_dim,
            use_bias=config.bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                small_init_initializer(dim=config.embedding_dim),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.proj_down = nnx.Linear(
            in_features=config._proj_up_dim,
            out_features=config.embedding_dim,
            use_bias=config.bias,
            rngs=rngs,
            param_dtype=dtype,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(
                wang_initializer(
                    dim=config.embedding_dim, num_blocks=config._num_blocks
                ),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.act_fn = get_act_fn(config.act_fn)
        self.dropout = nnx.Dropout(rate=config.dropout, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool = False):
        # Project up and split into gate and activation path
        up_proj_output = self.proj_up(x)
        gate_preact, up_proj = jnp.split(
            up_proj_output,
            indices_or_sections=2,
            # indices_or_sections=[self.config._proj_up_dim],
            axis=-1,
        )

        activated = self.act_fn(gate_preact) * up_proj
        output = self.proj_down(activated)
        output = self.dropout(output, deterministic=not training)

        return output


def create_feedforward(
    config: FeedForwardConfig, mesh: Mesh, rngs: nnx.Rngs, dtype=jnp.float32
):
    """Factory function to create feedforward modules based on config."""
    if config.ff_type == "ffn_gated":
        return GatedFeedForward(config, mesh=mesh, rngs=rngs, dtype=dtype)

    raise ValueError(f"Unknown feedforward type {config.ff_type}")
