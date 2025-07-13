# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from ...components.conv import CausalConv1d, CausalConv1dConfig
from ...components.init import small_init_initializer, wang_initializer
from ...components.linear_headwise import (
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from ...utils import UpProjConfigMixin
from .cell import mLSTMCell, mLSTMCellConfig


@dataclass(unsafe_hash=True, order=True)
class mLSTMLayerConfig(UpProjConfigMixin):
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    num_heads: int = 4
    proj_factor: float = 2.0

    # will be set toplevel config
    embedding_dim: int = -1
    bias: bool = False
    dropout: float = 0.0
    context_length: int = -1

    _num_blocks: int = 1
    _inner_embedding_dim: Optional[int] = None

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        self._inner_embedding_dim = self._proj_up_dim


class mLSTMLayer(nnx.Module):
    config_class = mLSTMLayerConfig

    def __init__(
        self,
        config: mLSTMLayerConfig,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        # Up-projection
        self.proj_up = nnx.Linear(
            in_features=config.embedding_dim,
            out_features=2 * config._inner_embedding_dim,
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

        # QKV projections
        num_proj_heads = round(config._inner_embedding_dim // config.qkv_proj_blocksize)
        qkv_config = LinearHeadwiseExpandConfig(
            in_features=config._inner_embedding_dim,
            num_heads=num_proj_heads,
            bias=config.bias,
        )

        self.q_proj = LinearHeadwiseExpand(
            config=qkv_config,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
        )

        self.k_proj = LinearHeadwiseExpand(
            config=qkv_config,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
        )

        self.v_proj = LinearHeadwiseExpand(
            config=qkv_config,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
        )

        # Convolutional layer
        self.conv1d = CausalConv1d(
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
            config=CausalConv1dConfig(
                feature_dim=config._inner_embedding_dim,
                kernel_size=config.conv1d_kernel_size,
            ),
        )

        # mLSTM cell
        self.mlstm_cell = mLSTMCell(
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
            config=mLSTMCellConfig(
                context_length=config.context_length,
                embedding_dim=config._inner_embedding_dim,
                num_heads=config.num_heads,
            ),
        )

        self.ogate_act_fn = jax.nn.swish

        # Learnable skip connection parameter
        self.learnable_skip = nnx.Param(
            jnp.empty(config._inner_embedding_dim, dtype=dtype),
            init_fn=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        # Down-projection
        self.proj_down = nnx.Linear(
            in_features=config._inner_embedding_dim,
            out_features=config.embedding_dim,
            use_bias=config.bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                wang_initializer(
                    dim=config.embedding_dim, num_blocks=config._num_blocks
                ),
                sharding=("tp", None),
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.dropout = nnx.Dropout(rate=config.dropout, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool = False):
        """Forward pass for processing a full sequence.

        Args:
            x: Input tensor of shape (B, S, H)

        Returns:
            Output tensor of shape (B, S, H)
        """

        # Up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = jnp.split(x_inner, indices_or_sections=2, axis=-1)

        # mLSTM branch
        x_mlstm_conv = self.conv1d(x_mlstm)
        x_mlstm_conv_act = jax.nn.swish(x_mlstm_conv)  # SiLU is same as swish

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)

        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # Output / z branch
        h_state = h_tilde_state_skip * self.ogate_act_fn(z)

        # Down-projection with dropout
        y = self.dropout(self.proj_down(h_state), deterministic=not training)
        return y
