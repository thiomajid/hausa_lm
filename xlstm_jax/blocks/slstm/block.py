# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbininan Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from ...components.feedforward import FeedForwardConfig
from ..xlstm_block import xLSTMBlock, xLSTMBlockConfig
from .layer import sLSTMLayerConfig


@dataclass(unsafe_hash=True, order=True)
class sLSTMBlockConfig:
    slstm: sLSTMLayerConfig = field(default_factory=sLSTMLayerConfig)
    feedforward: Optional[FeedForwardConfig] = field(default_factory=FeedForwardConfig)

    # we initialize these with None to catch the case where they are not set
    _num_blocks: Optional[int] = None
    _block_idx: Optional[int] = None

    def __post_init__(self):
        # Make sure block indexes are properly propagated to layer configs
        if hasattr(self.slstm, "_block_idx"):
            self.slstm._block_idx = self._block_idx
        if hasattr(self.slstm, "_num_blocks"):
            self.slstm._num_blocks = self._num_blocks

        # Trigger post_init for subconfigs
        if hasattr(self.slstm, "__post_init__"):
            self.slstm.__post_init__()

        if self.feedforward is not None and hasattr(self.feedforward, "__post_init__"):
            self.feedforward.__post_init__()


class sLSTMBlock(xLSTMBlock):
    """sLSTM block implementation based on the xLSTM building block architecture.

    This specialized block uses sLSTM layer and optionally a feedforward component.
    """

    config_class = sLSTMBlockConfig

    def __init__(
        self,
        config: sLSTMBlockConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        """Initialize an sLSTM block.

        Args:
            config: Configuration object for the sLSTM block
        """
        # Create an xLSTM config with only the sLSTM component active
        xlstm_config = xLSTMBlockConfig(
            mlstm=None,
            slstm=config.slstm,
            feedforward=config.feedforward,
            _block_idx=config._block_idx,
            _num_blocks=config._num_blocks,
        )

        # Initialize using the parent class constructor
        super().__init__(config=xlstm_config, mesh=mesh, rngs=rngs, dtype=dtype)
