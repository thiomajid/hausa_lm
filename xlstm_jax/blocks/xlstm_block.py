# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from ..components.feedforward import FeedForwardConfig, create_feedforward
from ..components.ln import LayerNorm
from .mlstm.layer import mLSTMLayer, mLSTMLayerConfig
from .slstm.layer import sLSTMLayer, sLSTMLayerConfig


@dataclass
class xLSTMBlockConfig:
    mlstm: Optional[mLSTMLayerConfig] = None
    slstm: Optional[sLSTMLayerConfig] = None
    feedforward: Optional[FeedForwardConfig] = None

    # we initialize these with None to catch the case where they are not set
    _num_blocks: Optional[int] = None
    _block_idx: Optional[int] = None

    def __post_init__(self):
        assert self.mlstm is not None or self.slstm is not None, (
            "Either mlstm or slstm must be provided"
        )
        assert self.mlstm is None or self.slstm is None, (
            "Only one of mlstm or slstm can be provided"
        )

        embedding_dim = (
            self.mlstm.embedding_dim
            if self.mlstm is not None
            else self.slstm.embedding_dim
        )

        if self.mlstm:
            self.mlstm._num_blocks = self._num_blocks
            self.mlstm._block_idx = self._block_idx

        if self.slstm:
            self.slstm._num_blocks = self._num_blocks
            self.slstm._block_idx = self._block_idx

        if self.feedforward:
            self.feedforward.embedding_dim = embedding_dim
            self.feedforward._num_blocks = self._num_blocks
            self.feedforward.__post_init__()


class xLSTMBlock(nnx.Module):
    """An xLSTM block can be either an sLSTM Block or an mLSTM Block.

    It contains the pre-LayerNorms and the skip connections.
    """

    config_class = xLSTMBlockConfig

    def __init__(
        self,
        config: xLSTMBlockConfig,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ) -> None:
        """Initialize an xLSTM block.

        Args:
            config: Configuration for the xLSTM block
        """

        embedding_dim = (
            config.mlstm.embedding_dim
            if config.mlstm is not None
            else config.slstm.embedding_dim
        )

        self.xlstm_norm: nnx.LayerNorm = LayerNorm(
            num_features=embedding_dim,
            use_scale=True,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            mesh=mesh,
        )

        if config.mlstm is not None:
            self.xlstm = mLSTMLayer(
                config=config.mlstm, mesh=mesh, rngs=rngs, dtype=dtype
            )

        elif config.slstm is not None:
            self.xlstm = sLSTMLayer(
                config=config.slstm, mesh=mesh, rngs=rngs, dtype=dtype
            )
        else:
            raise ValueError("Either mlstm or slstm must be provided")

        if config.feedforward is not None:
            self.ffn_norm = LayerNorm(
                num_features=config.feedforward.embedding_dim,
                use_scale=True,
                use_bias=False,
                rngs=rngs,
                dtype=dtype,
                mesh=mesh,
            )

            self.ffn = create_feedforward(
                config=config.feedforward,
                mesh=mesh,
                rngs=rngs,
                dtype=dtype,
            )
        else:
            self.ffn_norm = None
            self.ffn = None

    def __call__(self, x: jax.Array):
        """Process a full sequence through the xLSTM block.

        Args:
            x: Input tensor of shape (B, S, D)

        Returns:
            Output tensor of shape (B, S, D)
        """
        x_normed = self.xlstm_norm(x)
        x_xlstm = self.xlstm(x_normed)
        x = x + x_xlstm

        # can't use lax.cond here because when evaluated, on a branch, the
        # ffn is None, so it will not be called
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x))

        return x
