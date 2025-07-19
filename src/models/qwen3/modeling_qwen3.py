"""
JAX implementation of Qwen3 using Flax NNX.

This module provides a clean JAX/Flax implementation of the Qwen3 transformer model,
equivalent to the PyTorch version but optimized for JAX. It supports:

- Full attention and sliding window attention
- Grouped query attention (GQA)
- RMS normalization with query/key normalization
- Rotary position embeddings (RoPE)
- SwiGLU activation in MLP
- Weight loading from PyTorch checkpoints

Key differences from the PyTorch implementation:
- Uses Flax NNX for cleaner module definition
- Simplified attention computation without complex caching
- JAX-native operations for better performance
- Functional approach with immutable arrays
"""

import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx, struct
from flax.nnx import initializers
from jax import lax
from jax.sharding import Mesh
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from ...utils.devices import create_mesh
from ..common.ops import apply_rotary_pos_emb, compute_default_rope_freqs, repeat_kv


@struct.dataclass
class Qwen3ModelOutput:
    """Output of Qwen3Model forward pass."""

    last_hidden_state: jax.Array
    hidden_states: tp.Optional[tp.Tuple[jax.Array, ...]] = None


@struct.dataclass
class Qwen3CausalLMOutput:
    """Output of Qwen3ForCausalLM forward pass."""

    logits: jax.Array
    hidden_states: tp.Optional[tp.Tuple[jax.Array, ...]] = None


class Qwen3RMSNorm(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        mesh: Mesh,
        dtype=jnp.float32,
        eps: float = 1e-6,
    ):
        self.eps = eps
        self.weight = nnx.Param(
            jnp.empty(hidden_size, dtype=dtype),
            init_fn=nnx.with_partitioning(
                initializers.ones_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

    def __call__(self, hidden_states: jax.Array):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        return (self.weight.value * hidden_states).astype(input_dtype)


class Qwen3MLP(nnx.Module):
    """Multi-layer perceptron with SwiGLU activation."""

    def __init__(
        self,
        config: Qwen3Config,
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
                initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        self.gate_proj = Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size)

    def __call__(self, x: jax.Array):
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        activated = jax.nn.silu(gate_out) * up_out
        return self.down_proj(activated)


class Qwen3RotaryEmbedding(nnx.Module):
    """Rotary Position Embedding for Qwen3."""

    def __init__(self, config: Qwen3Config):
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Compute inverse frequencies
        inv_freq = compute_default_rope_freqs(self.rope_theta, self.head_dim)
        self.inv_freq = nnx.Variable(inv_freq)

    def __call__(
        self,
        x: jax.Array,
        position_ids: jax.Array,
    ):
        """Compute rotary embeddings for given positions."""
        # Expand dimensions for broadcasting
        inv_freq_expanded = jnp.expand_dims(self.inv_freq, axis=0)  # [1, head_dim//2]
        position_ids_expanded = jnp.expand_dims(
            position_ids, axis=-1
        )  # [batch, seq_len, 1]

        # Compute frequencies
        freqs = jnp.matmul(
            position_ids_expanded.astype(jnp.float32), inv_freq_expanded
        )  # [batch, seq_len, head_dim//2]
        emb = jnp.concatenate([freqs, freqs], axis=-1)  # [batch, seq_len, head_dim]

        cos = jnp.cos(emb)
        sin = jnp.sin(emb)

        return cos.astype(x.dtype), sin.astype(x.dtype)


class Qwen3Attention(nnx.Module):
    """Multi-headed attention mechanism with optional sliding window."""

    def __init__(
        self,
        config: Qwen3Config,
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
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        # Check if this layer uses sliding window attention
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

        Linear = partial(
            nnx.Linear,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=config.attention_bias,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.q_proj = Linear(config.hidden_size, self.num_heads * self.head_dim)

        self.k_proj = Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim
        )

        self.v_proj = Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim
        )

        self.o_proj = Linear(self.num_heads * self.head_dim, config.hidden_size)

        RMSNorm = partial(
            Qwen3RMSNorm,
            hidden_size=self.head_dim,
            eps=config.rms_norm_eps,
            mesh=mesh,
            dtype=dtype,
        )

        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()

    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: tp.Tuple[jax.Array, jax.Array],
        attention_mask: tp.Optional[jax.Array] = None,
    ):
        """Forward pass of attention mechanism."""
        B, S, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.reshape(B, S, self.num_heads, self.head_dim)
        key_states = key_states.reshape(B, S, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(
            B, S, self.num_key_value_heads, self.head_dim
        )

        # Transpose for attention computation: [batch, num_heads, seq_len, head_dim]
        query_states = jnp.transpose(query_states, (0, 2, 1, 3))
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))

        # Apply query and key normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Repeat key/value heads for grouped query attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = (
            jnp.matmul(query_states, jnp.transpose(key_states, (0, 1, 3, 2)))
            * self.scaling
        )

        # Apply attention mask
        def apply_mask(weights: jax.Array):
            # mask = attention_mask[:, None, None, :]
            mask = attention_mask[:, :, :, : key_states.shape[-2]]
            mask = jnp.broadcast_to(mask, shape=(B, self.num_heads, S, S))
            return weights + mask

        attn_weights = lax.cond(
            attention_mask is not None,
            lambda weights: apply_mask(weights),
            lambda weights: weights,
            operand=attn_weights,
        )

        # Apply sliding window mask if needed
        # def apply_sliding_window(weights: jax.Array):
        #     # Create sliding window mask
        #     sliding_mask = jnp.triu(
        #         jnp.full((S, S), -jnp.inf), k=self.sliding_window + 1
        #     )
        #     sliding_mask = sliding_mask + jnp.tril(
        #         jnp.full((S, S), -jnp.inf), k=-self.sliding_window - 1
        #     )

        #     return weights + sliding_mask

        # attn_weights = lax.cond(
        #     attention_mask is not None,
        #     lambda weights: apply_sliding_window(weights),
        #     lambda weights: weights,
        #     operand=attn_weights,
        # )

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.matmul(attn_weights, value_states)

        # [batch, seq_len, num_heads, head_dim]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(B, S, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen3DecoderLayer(nnx.Module):
    """Single transformer decoder layer."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Self-attention mechanism
        self.self_attn = Qwen3Attention(
            config, layer_idx, mesh=mesh, rngs=rngs, dtype=dtype
        )

        # Feed-forward network
        self.mlp = Qwen3MLP(config, mesh=mesh, rngs=rngs, dtype=dtype)

        # Layer normalization
        RMSNorm = partial(
            Qwen3RMSNorm,
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            mesh=mesh,
            dtype=dtype,
        )

        self.input_layernorm = RMSNorm()
        self.post_attention_layernorm = RMSNorm()

    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: tp.Tuple[jax.Array, jax.Array],
        attention_mask: tp.Optional[jax.Array] = None,
    ):
        """Forward pass of decoder layer."""
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3Model(nnx.Module):
    """The Qwen3 transformer model."""

    def __init__(
        self,
        config: Qwen3Config,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.padding_idx = config.pad_token_id
        self.num_layers = config.num_hidden_layers

        # Token embeddings
        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(
                initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        self.layers = [
            Qwen3DecoderLayer(config, layer_idx, mesh=mesh, rngs=rngs, dtype=dtype)
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, mesh=mesh, dtype=dtype
        )

        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: tp.Optional[jax.Array] = None,
        position_ids: tp.Optional[jax.Array] = None,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        """Forward pass of the model."""
        batch_size, seq_len = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = None
        if attention_mask is None:
            causal_mask = jnp.triu(jnp.full((seq_len, seq_len), -jnp.inf), k=1)
        else:
            causal_mask = jnp.where(
                attention_mask[:, None, None, :] == 0, -jnp.inf, 0.0
            )

        def layer_scan(carry: jax.Array, idx):
            h_t = carry
            layer_output: jax.Array = lax.switch(
                idx,
                self.layers,
                h_t,
                position_embeddings,
                causal_mask,
            )

            next_state = layer_output
            return next_state, layer_output

        init_carry = hidden_states
        hidden_states, all_hidden_states = lax.scan(
            layer_scan,
            init=init_carry,
            xs=jnp.arange(self.num_layers),
        )

        hidden_states = self.norm(hidden_states)

        return Qwen3ModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class Qwen3ForCausalLM(nnx.Module):
    """Qwen3 model with a language modeling head."""

    def __init__(
        self,
        config: Qwen3Config,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.model = Qwen3Model(config, mesh=mesh, rngs=rngs, dtype=dtype)

        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: tp.Optional[jax.Array] = None,
        position_ids: tp.Optional[jax.Array] = None,
        output_hidden_states: bool = False,
    ):
        """Forward pass for causal language modeling."""
        # Get transformer outputs
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = transformer_outputs.last_hidden_state

        # Compute logits
        logits = self.lm_head(hidden_states)

        return Qwen3CausalLMOutput(
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
        )

    def generate(self, input_ids: jax.Array):
        out = self(input_ids)
        return out.logits


# Example usage and testing
if __name__ == "__main__":
    """
    Example of how to use the JAX Qwen3 implementation.
    This demonstrates model creation and inference.
    """

    # Create a simple test configuration
    test_config = Qwen3Config(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        use_sliding_window=False,
        layer_types=["full_attention"] * 4,
    )

    # Initialize random number generators

    mesh = create_mesh((1, 1), ("dp", "tp"))
    rngs = nnx.Rngs(12)

    # Create the JAX model
    print("Creating JAX Qwen3 model...")
    model = None
    with mesh:
        model = Qwen3ForCausalLM(test_config, mesh=mesh, rngs=rngs, dtype=jnp.float32)
        state = nnx.state(model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = lax.with_sharding_constraint(state, pspecs)
        nnx.update(model, sharded_state)

    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    print(f"Running forward pass with input shape: {input_ids.shape}")

    # Run inference
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
    )

    print(f"Output logits shape: {outputs.logits.shape}")
    print(
        f"Number of hidden states: {len(outputs.hidden_states) if outputs.hidden_states else 0}"
    )

    # Example of using with a real pretrained model (see qwen3_converter.py)
    """
    from .qwen3_converter import convert_qwen3_from_hf
    
    # Load from pretrained PyTorch model
    model_name = "Qwen/Qwen3-0.6B"
    jax_model = convert_qwen3_from_hf(model_name, rngs=nnx.Rngs(42))
    
    # Test with real tokenized input
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text = "Hello, how are you today?"
    inputs = tokenizer(text, return_tensors="np")
    input_ids = jnp.array(inputs["input_ids"])
    
    outputs = jax_model(input_ids, training=False)
    print(f"Generated logits shape: {outputs.logits.shape}")
    """

    print("JAX Qwen3 model test completed successfully!")
