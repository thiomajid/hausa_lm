import typing as tp

import jax
import jax.numpy as jnp


def rotate_half(x: jax.Array) -> jax.Array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: jax.Array,
    k: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> tp.Tuple[jax.Array, jax.Array]:
    """Apply rotary position embedding to query and key tensors."""
    cos = jnp.expand_dims(cos, axis=1)  # [batch, 1, seq_len, head_dim]
    sin = jnp.expand_dims(sin, axis=1)  # [batch, 1, seq_len, head_dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: jax.Array, n_rep: int) -> jax.Array:
    """Repeat key/value heads for grouped query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    # Expand and reshape to repeat heads
    hidden_states = jnp.expand_dims(
        hidden_states, axis=2
    )  # [batch, num_kv_heads, 1, slen, head_dim]
    hidden_states = jnp.repeat(
        hidden_states, n_rep, axis=2
    )  # [batch, num_kv_heads, n_rep, slen, head_dim]
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def compute_default_rope_freqs(rope_theta: float, head_dim: int):
    return 1.0 / (
        rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
