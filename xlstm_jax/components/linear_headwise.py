# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbininan Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from xlstm_jax.components.init import small_init_initializer


@dataclass(unsafe_hash=True, order=True)
class LinearHeadwiseExpandConfig:
    in_features: int = 0
    # this is the number of heads that the in_features are split into
    # if num_heads=1, this is a normal linear layer
    # if num_heads>1, the in_features are split into num_heads and each head is projected separately
    # if num_heads=in_features, each feature is projected separately
    num_heads: int = -1
    expand_factor_up: float = 1

    # this is internally computed
    # but can be overwritten if you want to use a different output dimension
    # if > 0 the expand factor is ignored
    _out_features: int = -1

    bias: bool = True
    trainable_weight: bool = True
    trainable_bias: bool = True

    def __post_init__(self):
        assert self.num_heads > 0, "num_heads must be set"
        assert self.num_heads <= self.in_features, "num_heads must be <= in_features"
        assert self.in_features % self.num_heads == 0, (
            "in_features must be a multiple of num_heads"
        )

        if self._out_features < 0:
            self._out_features = round(self.expand_factor_up * self.in_features)


class LinearHeadwiseExpand(nnx.Module):
    """This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    config_class = LinearHeadwiseExpandConfig

    def __init__(
        self,
        config: LinearHeadwiseExpandConfig,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        in_features = config.in_features
        num_heads = config.num_heads
        in_features_per_head = in_features // num_heads
        out_features_per_head = config._out_features // num_heads

        self.trainable_kernel = config.trainable_weight
        self.trainable_bias = config.trainable_bias
        self.num_heads = num_heads

        # Create weight parameter
        self.kernel = nnx.Param(
            jnp.empty(
                (num_heads, out_features_per_head, in_features_per_head),
                dtype=dtype,
            ),
            # init_fn=jax.nn.initializers.normal(stddev=stddev),
            init_fn=nnx.with_partitioning(
                small_init_initializer(in_features_per_head),
                sharding=(None, None, "tp"),
                mesh=mesh,
            ),
        )

        # Create bias parameter
        if config.bias:
            self.bias = nnx.Param(
                jnp.zeros(config._out_features, dtype=dtype),
                init_fn=nnx.with_partitioning(
                    nnx.initializers.zeros_init(),
                    sharding=("tp",),
                    mesh=mesh,
                ),
            )
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass for the headwise linear transformation.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        shape = x.shape
        x = x.reshape(*shape[:-1], self.num_heads, -1)
        kernel = jax.lax.cond(
            self.trainable_kernel,
            lambda: self.kernel.value,
            lambda: jax.lax.stop_gradient(self.kernel.value),
        )

        x = jnp.einsum("...hd,hod->...ho", x, kernel)
        x = x.reshape(*shape[:-1], -1)

        if self.bias is not None:
            bias = jax.lax.cond(
                self.trainable_bias,
                lambda: self.bias.value,
                lambda: jax.lax.stop_gradient(self.bias.value),
            )

            x = x + bias

        return x
