# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbininan Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
import typing as tp
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from ...components.conv import CausalConv1d, CausalConv1dConfig
from ...components.linear_headwise import (
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from ...components.ln import MultiHeadLayerNorm
from .cell import sLSTMCell, sLSTMCellConfig


@dataclass(unsafe_hash=True, order=True)
class sLSTMLayerConfig(sLSTMCellConfig):
    embedding_dim: int = -1
    num_heads: int = 4  # this must divide the hidden size, is not yet supported by all versions in this directory
    conv1d_kernel_size: int = 4  # 0 means no convolution included
    group_norm_weight: bool = True
    dropout: float = 0.0

    def __post_init__(self):
        self.hidden_size = self.embedding_dim
        sLSTMCellConfig.__post_init__(self)


class sLSTMLayer(nnx.Module):
    """sLSTM layer implementation in JAX/Flax."""

    config_class = sLSTMLayerConfig

    def __init__(
        self,
        config: sLSTMLayerConfig,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.conv1d_kernel_size = config.conv1d_kernel_size

        # Initialize convolutional layer if needed
        if config.conv1d_kernel_size > 0:
            self.conv1d = CausalConv1d(
                rngs=rngs,
                dtype=dtype,
                mesh=mesh,
                config=CausalConv1dConfig(
                    feature_dim=config.embedding_dim,
                    kernel_size=config.conv1d_kernel_size,
                ),
            )

        # Initialize gate projections using headwise linear layers
        gate_config = LinearHeadwiseExpandConfig(
            in_features=config.embedding_dim,
            num_heads=config.num_heads,
            bias=False,
        )

        # We initialize all gate projections explicitly
        self.fgate = LinearHeadwiseExpand(
            config=gate_config,
            rngs=rngs,
            dtype=dtype,
            mesh=mesh,
        )

        self.igate = LinearHeadwiseExpand(
            config=gate_config,
            rngs=rngs,
            dtype=dtype,
            mesh=mesh,
        )

        self.zgate = LinearHeadwiseExpand(
            config=gate_config,
            rngs=rngs,
            dtype=dtype,
            mesh=mesh,
        )

        self.ogate = LinearHeadwiseExpand(
            config=gate_config,
            rngs=rngs,
            dtype=dtype,
            mesh=mesh,
        )

        # sLSTM cell and normalization
        self.slstm_cell = sLSTMCell(config, mesh=mesh, rngs=rngs, dtype=dtype)
        self.group_norm = MultiHeadLayerNorm(
            num_features=config.embedding_dim,
            use_scale=config.group_norm_weight,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.dropout = nnx.Dropout(rate=config.dropout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        slstm_state: tp.Optional[jax.Array] = None,
        return_last_state: bool = False,
        training: bool = False,
    ) -> tp.Union[jax.Array, tuple[jax.Array, dict[str, tp.Any]]]:
        """Process a sequence through the sLSTM layer.

        Args:
            x: Input tensor of shape (B, S, D)
            conv_state: Initial convolution state
            slstm_state: Initial sLSTM cell state
            return_last_state: Whether to return final states

        Returns:
            Output tensor or tuple of output tensor and final states
        """
        B, S, _ = x.shape

        x_conv = jax.lax.cond(
            self.conv1d_kernel_size > 0,
            lambda _x: jax.nn.swish(self.conv1d(_x)),
            lambda _x: _x,
            operand=x,
        )

        # Apply gate projections
        f = self.fgate(x_conv)
        i = self.igate(x_conv)
        z = self.zgate(x)
        o = self.ogate(x)

        # Concatenate gate outputs into a single tensor
        gates_combined = jnp.concatenate([i, f, z, o], axis=-1)

        # Process through sLSTM cell
        y, slstm_state = self.slstm_cell(gates_combined, state=slstm_state)
        y = self.dropout(y, deterministic=not training)

        out = self.group_norm(y)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)

        if return_last_state:
            return out, {"slstm_state": slstm_state}
        else:
            return out
