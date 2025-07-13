"""
NNX BERT model.
Code adapted from transformers.models.bert.modeling_bert.py
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import BertConfig, SigLip

from src.utils.initialization import (
    embedding_partitioned_init,
    linear_bias_partitioned_init,
    linear_kernel_partitioned_init,
    norm_bias_partitioned_init,
    norm_scale_partitioned_init,
)
from src.utils.mask import apply_padding_mask_with_gradient_stop, create_padding_mask


class BertEmbeddings(nnx.Module):
    def __init__(
        self,
        config: BertConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        Embed = partial(
            nnx.Embed,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
            embedding_init=embedding_partitioned_init(
                mesh,
                sharding=(None, "model"),
            ),
        )

        self.word_embeddings = Embed(num_embeddings=config.vocab_size)
        self.position_embeddings = Embed(config.max_position_embeddings)
        self.token_type_embeddings = Embed(config.type_vocab_size)

        self.LayerNorm = nnx.LayerNorm(
            num_features=config.hidden_size,
            epsilon=config.layer_norm_eps,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=norm_scale_partitioned_init(mesh),
            bias_init=norm_bias_partitioned_init(mesh),
        )

        self.dropout = nnx.Dropout(config.hidden_dropout_prob, rngs=rngs)

        # position_ids buffer
        position_ids = jnp.arange(config.max_position_embeddings, dtype=dtype)
        position_ids = jnp.broadcast_to(
            position_ids,
            shape=(1, -1),
            out_sharding=NamedSharding(mesh, P(None, "model")),
        )

        self.position_ids = nnx.Variable(position_ids)

        # token_type_ids buffer
        token_type_ids = jnp.zeros(
            shape=position_ids.shape,
            dtype=jnp.int32,
            device=NamedSharding(mesh, P(None, "model")),
        )

        self.token_type_ids = nnx.Variable(token_type_ids)

        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.pad_token_idx = config.pad_token_id

    def __call__(
        self,
        input_ids: jax.Array,
        past_key_values_length: int = 0,
        training: bool = False,
    ):
        seq_length = input_ids.shape[1]
        position_ids = self.position_ids.value[
            :, past_key_values_length : seq_length + past_key_values_length
        ]

        buffered_token_type_ids = self.token_type_ids.value[:, :seq_length]
        buffered_token_type_ids_expanded = jnp.broadcast_to(
            buffered_token_type_ids,
            shape=(input_ids.shape[0], seq_length),
        )

        token_type_ids = buffered_token_type_ids_expanded

        inputs_embeds = self.word_embeddings(input_ids)
        padding_mask = create_padding_mask(input_ids, self.pad_token_idx)
        inputs_embeds = apply_padding_mask_with_gradient_stop(
            inputs_embeds, padding_mask
        )

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        embeddings = lax.cond(
            self.position_embedding_type == "absolute",
            lambda x: x + self.position_embeddings(position_ids),
            lambda x: x,
            operand=embeddings,
        )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, deterministic=not training)
        return embeddings


class BertSelfOutput(nnx.Module):
    def __init__(
        self,
        config: BertConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.dense = nnx.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=linear_kernel_partitioned_init(
                mesh=mesh,
                sharding=(None, "model"),
            ),
            bias_init=linear_bias_partitioned_init(
                mesh=mesh,
                sharding=("model",),
            ),
        )

        self.LayerNorm = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=norm_scale_partitioned_init(mesh),
            bias_init=norm_bias_partitioned_init(mesh),
        )

        self.dropout = nnx.Dropout(config.hidden_dropout_prob, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        input_tensor: jax.Array,
        training: bool = False,
    ):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=not training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nnx.Module):
    def __init__(
        self,
        config: BertConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.dense = nnx.Linear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=linear_kernel_partitioned_init(
                mesh=mesh,
                sharding=(None, "model"),
            ),
            bias_init=linear_bias_partitioned_init(
                mesh=mesh,
                sharding=("model",),
            ),
        )

        self.activation = jax.nn.gelu

    def __call__(self, hidden_states: jax.Array):
        hidden_states = self.activation(self.dense(hidden_states))
        return hidden_states


class BertOutput(nnx.Module):
    def __init__(
        self,
        config: BertConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.dense = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=linear_kernel_partitioned_init(
                mesh=mesh,
                sharding=(None, "model"),
            ),
            bias_init=linear_bias_partitioned_init(
                mesh=mesh,
                sharding=("model",),
            ),
        )

        self.LayerNorm = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=norm_scale_partitioned_init(mesh),
            bias_init=norm_bias_partitioned_init(mesh),
        )

        self.dropout = nnx.Dropout(config.hidden_dropout_prob)

    def __call__(
        self,
        hidden_states: jax.Array,
        input_tensor: jax.Array,
        training: bool = False,
    ):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=not training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
