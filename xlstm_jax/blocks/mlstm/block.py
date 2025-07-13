# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from ..xlstm_block import xLSTMBlock, xLSTMBlockConfig
from .layer import mLSTMLayerConfig


@dataclass(unsafe_hash=True, order=True)
class mLSTMBlockConfig:
    mlstm: mLSTMLayerConfig = field(default_factory=mLSTMLayerConfig)

    # we initialize these with None to catch the case where they are not set
    _num_blocks: Optional[int] = None
    _block_idx: Optional[int] = None

    def __post_init__(self):
        if self._num_blocks is not None:
            self.mlstm._num_blocks = self._num_blocks
        self.mlstm.__post_init__()


class mLSTMBlock(xLSTMBlock):
    """mLSTM block implementation based on the xLSTM building block architecture.

    This is a simplified block that only uses the mLSTM layer component
    and not the sLSTM or feedforward components.
    """

    config_class = mLSTMBlockConfig

    def __init__(
        self,
        config: mLSTMBlockConfig,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ) -> None:
        """Initialize an mLSTM block.

        Args:
            config: Configuration object for the mLSTM block
        """

        xlstm_config = xLSTMBlockConfig(
            mlstm=config.mlstm,
            slstm=None,
            feedforward=None,
            _num_blocks=config._num_blocks,
            _block_idx=config._block_idx,
        )

        # Initialize using the parent class constructor
        super().__init__(config=xlstm_config, mesh=mesh, rngs=rngs, dtype=dtype)
