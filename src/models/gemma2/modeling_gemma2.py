import math
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import Mesh
from transformers import Gemma2Config

from xlstm_jax.mask import apply_padding_mask_with_gradient_stop, create_padding_mask


def rotate_half(x: jax.Array):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: jax.Array,
    k: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    unsqueeze_dim: int = 1,
):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@partial(jax.jit, static_argnames=("n_rep",))
def repeat_kv(hidden_states: jax.Array, n_rep: int):
    """Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.expand_dims(hidden_states, axis=2)
    hidden_states = jnp.tile(hidden_states, (1, 1, n_rep, 1, 1))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@partial(jax.jit, static_argnums=(1,))
def soft_cap(weights: jax.Array, cap):
    weights = weights / cap
    weights = jnp.tanh(weights)
    return weights * cap


class Gemma2RMSNorm(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
        eps: float = 1e-6,
        scale_init=nnx.initializers.ones_init(),
    ):
        self.eps = eps
        self.dtype = dtype

        self.scale = nnx.Param(scale_init(rngs.params(), (dim,), dtype))

    def __call__(self, x: jax.Array):
        output = self._norm(x.astype(self.dtype))
        output = output * (1.0 + self.scale)
        return output.astype(x.dtype)

    def _norm(self, x: jax.Array):
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)


class Gemma2MLP(nnx.Module):
    def __init__(
        self,
        config: Gemma2Config,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        Linear = partial(
            nnx.Linear,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        self.gate_proj = Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size)

        self.act_fn = jax.nn.gelu

    def __call__(self, x: jax.Array):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down


class Gemma2Attention(nnx.Module):
    def __init__(
        self,
        config: Gemma2Config,
        layer_idx: int,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        Linear = partial(
            nnx.Linear,
            use_bias=config.attention_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.q_proj = Linear(
            in_features=config.hidden_size,
            out_features=config.num_attention_heads * self.head_dim,
        )

        self.k_proj = Linear(
            in_features=config.hidden_size,
            out_features=config.num_key_value_heads * self.head_dim,
        )

        self.v_proj = Linear(
            in_features=config.hidden_size,
            out_features=config.num_key_value_heads * self.head_dim,
        )

        self.o_proj = Linear(
            in_features=config.num_attention_heads * self.head_dim,
            out_features=config.hidden_size,
        )

        self.dropout = nnx.Dropout(config.attention_dropout)

        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if not bool(layer_idx % 2) else None

    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: tp.Tuple[jax.Array, jax.Array],
        attention_mask: tp.Optional[jax.Array] = None,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        B, S, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).reshape(hidden_shape)
        key_states = self.k_proj(hidden_states).reshape(hidden_shape)
        value_states = self.v_proj(hidden_states).reshape(hidden_shape)

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            jnp.matmul(query_states, key_states.transpose(0, 1, 3, 2)) * self.scaling
        )

        attn_weights = lax.cond(
            self.attn_logit_softcapping is not None,
            lambda _x: soft_cap(_x, self.attn_logit_softcapping),
            lambda _x: _x,
            operand=attn_weights,
        )

        # if attention_mask is not None:
        #     attn_weights = attn_weights + attention_mask

        def _apply_mask(weights: jax.Array):
            mask = attention_mask[:, None, None, :]
            mask = jnp.broadcast_to(mask, shape=(B, self.num_heads, S, S))
            return weights * mask

        attn_weights = lax.cond(
            attention_mask is not None,
            lambda weights: _apply_mask(weights),
            lambda weights: weights,
            operand=attn_weights,
        )

        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(
            query_states.dtype
        )

        attn_weights = self.dropout(attn_weights)
        attn_output = jnp.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, S, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Gemma2DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: Gemma2Config,
        layer_idx: int,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.hidden_size = config.hidden_size
        self.is_sliding = not bool(layer_idx % 2)

        self.self_attn = Gemma2Attention(
            config=config,
            layer_idx=layer_idx,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
        )

        self.mlp = Gemma2MLP(config, mesh=mesh, rngs=rngs, dtype=dtype)

        RMSNorm = partial(
            Gemma2RMSNorm,
            config.hidden_size,
            eps=config.rms_norm_eps,
            rngs=rngs,
            dtype=dtype,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.input_layernorm = RMSNorm()
        self.post_attention_layernorm = RMSNorm()
        self.pre_feedforward_layernorm = RMSNorm()
        self.post_feedforward_layernorm = RMSNorm()

        self.sliding_window = config.sliding_window

    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: tp.Tuple[jax.Array, jax.Array],
        attention_mask: tp.Optional[jax.Array] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights


class Gemma2RotaryEmbedding(nnx.Module):
    def __init__(self, config: Gemma2Config):
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        # Create inverse frequencies
        inv_freq = 1.0 / (
            self.rope_theta
            ** (jnp.arange(0, self.head_dim, 2, dtype=jnp.float32) / self.head_dim)
        )
        self.inv_freq = inv_freq

        # Attention scaling for Gemma2
        self.attention_scaling = 1.0 / math.sqrt(config.query_pre_attn_scalar)

    def __call__(self, x, position_ids):
        batch_size, seq_len = position_ids.shape
        inv_freq_expanded = self.inv_freq[None, :, None]
        inv_freq_expanded = jnp.broadcast_to(
            inv_freq_expanded, (batch_size, len(self.inv_freq), 1)
        )
        position_ids_expanded = position_ids[:, None, :].astype(jnp.float32)

        freqs = jnp.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)


class Gemma2Model(nnx.Module):
    def __init__(
        self,
        config: Gemma2Config,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=config.initializer_range),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
        )

        self.layers = [
            Gemma2DecoderLayer(config, layer_idx, mesh=mesh, rngs=rngs, dtype=dtype)
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.num_layers = len(self.layers)

        self.norm = Gemma2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            rngs=rngs,
            dtype=dtype,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )
        self.rotary_emb = Gemma2RotaryEmbedding(config=config)

    def __call__(self, input_ids: jax.Array, attention_mask: jax.Array):
        inputs_embeds = self.embed_tokens(input_ids)
        padding_mask = create_padding_mask(input_ids, self.padding_idx)
        inputs_embeds = apply_padding_mask_with_gradient_stop(
            inputs_embeds, padding_mask
        )

        batch_size, seq_length = inputs_embeds.shape[:2]

        position_ids = jnp.arange(seq_length)[None, :]
        position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_length))

        # Get rotary embeddings
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Apply layers
        def _layer_scan(carry: jax.Array, idx: jax.Array):
            h_t, _ = lax.switch(
                idx,
                self.layers,
                carry,
                position_embeddings,
                attention_mask,
            )

            return h_t, h_t

        h_t, layers_states = lax.scan(
            f=_layer_scan,
            init=inputs_embeds,
            xs=jnp.arange(self.num_layers),
        )

        h_t = self.norm(h_t)

        return h_t, layers_states


class Gemma2ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: Gemma2Config,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.config = config
        self.model = Gemma2Model(config, mesh=mesh, rngs=rngs, dtype=dtype)

        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        self.final_logit_softcapping = config.final_logit_softcapping

    def __call__(self, input_ids: jax.Array, attention_mask: jax.Array):
        hidden_states, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = self.lm_head(hidden_states)
        logits = soft_cap(logits, self.final_logit_softcapping)

        return logits
