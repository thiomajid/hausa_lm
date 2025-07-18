# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from xlstm_jax.mask import apply_padding_mask_with_gradient_stop, create_padding_mask

from .components.init import small_init_initializer
from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig


@dataclass(unsafe_hash=True, order=True)
class xLSTMLMModelConfig(xLSTMBlockStackConfig):
    vocab_size: int = -1
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False
    pad_token_id: int = 1


class xLSTMLMModel(nnx.Module):
    """Language model using xLSTM blocks as its backbone.

    This model combines token embeddings with an xLSTM block stack
    and a language modeling head for next token prediction.
    """

    config_class = xLSTMLMModelConfig

    def __init__(
        self,
        config: xLSTMLMModelConfig,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.xlstm_block_stack = xLSTMBlockStack(
            config=config,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
        )

        self.token_embedding = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.embedding_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        self.embedding_dropout = (
            nnx.Dropout(rate=config.dropout, rngs=rngs)
            if config.add_embedding_dropout
            else jax.nn.identity
        )

        self.lm_head = nnx.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
            use_bias=False,
            rngs=rngs,
            param_dtype=dtype,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        # Create shared embedding parameters if using weight tying
        if config.tie_weights:
            # Create a single shared weight for both embedding and output
            self.shared_weight = nnx.Param(
                jnp.zeros((config.vocab_size, config.embedding_dim), dtype=dtype),
                init_fn=nnx.with_partitioning(
                    small_init_initializer(dim=config.embedding_dim),
                    sharding=(None, "tp"),
                    mesh=mesh,
                ),
            )
        else:
            self.shared_weight = None

        self.tie_weights = config.tie_weights
        self.pad_token_id = config.pad_token_id

    def __call__(self, input_ids: jax.Array):
        """Forward pass through the model.

        Args:
            idx: Input token indices of shape [B, S]

        Returns:
            Logits of shape [B, S, vocab_size]
        """

        # Get embedding weights (either shared or dedicated)
        h_t = None
        if self.tie_weights:
            emb_weight = self.shared_weight
            h_t = jnp.take(emb_weight, input_ids, axis=0)
        else:
            h_t = self.token_embedding(input_ids)

        padding_mask = create_padding_mask(input_ids, self.pad_token_id)
        h_t = apply_padding_mask_with_gradient_stop(h_t, padding_mask)

        h_t = self.embedding_dropout(h_t)
        h_t, _ = self.xlstm_block_stack(h_t)

        # Apply language model head
        logits = None
        if self.tie_weights:
            logits = jnp.matmul(h_t, self.shared_weight.T)
        else:
            logits = self.lm_head(h_t)

        return logits

    def generate(self, input_ids: jax.Array):
        return self(input_ids)
