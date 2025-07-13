# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from ...components.init import bias_linspace_initializer
from ...components.ln import MultiHeadLayerNorm
from .backends import parallel_stabilized_simple, recurrent_step_stabilized_simple


@dataclass
class mLSTMCellConfig:
    context_length: int = -1
    embedding_dim: int = -1
    num_heads: int = -1


class mLSTMCell(nnx.Module):
    config_class = mLSTMCellConfig

    def __init__(
        self,
        config: mLSTMCellConfig,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        # Store configuration parameters for easier access
        self.context_length = config.context_length
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.dim_per_head = self.embedding_dim // self.num_heads

        # Set backend functions
        self.backend_fn = parallel_stabilized_simple
        self.backend_fn_step = recurrent_step_stabilized_simple

        # Gate projections
        self.igate = nnx.Linear(
            in_features=3 * config.embedding_dim,
            out_features=config.num_heads,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.1, dtype=dtype),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.fgate = nnx.Linear(
            in_features=3 * config.embedding_dim,
            out_features=config.num_heads,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                bias_linspace_initializer(start=3.0, end=6.0),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        # Output normalization
        self.outnorm = MultiHeadLayerNorm(
            num_features=config.embedding_dim,
            use_scale=True,
            use_bias=False,
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

        self.causal_mask = nnx.Variable(
            jnp.tril(
                jnp.ones(
                    (config.context_length, config.context_length),
                    dtype=jnp.bool,
                )
            )
        )

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
    ) -> jax.Array:
        """Forward pass for parallel processing of the entire sequence.

        Args:
            q: Query tensor of shape (B, S, H)
            k: Key tensor of shape (B, S, H)
            v: Value tensor of shape (B, S, H)

        Returns:
            Output tensor of shape (B, S, H)
        """
        B, S, _ = q.shape  # (B, S, H)

        # Combine inputs for gate computation
        if_gate_input = jnp.concatenate([q, k, v], axis=-1)

        # Compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = jnp.transpose(igate_preact, (0, 2, 1))
        igate_preact = igate_preact[..., None]  # (B, NH, S, 1)

        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = jnp.transpose(fgate_preact, (0, 2, 1))
        fgate_preact = fgate_preact[..., None]  # (B, NH, S, 1)

        # Reshape for multi-head processing
        q_reshaped = q.reshape(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k_reshaped = k.reshape(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v_reshaped = v.reshape(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        # Transpose to put heads dimension before sequence dimension
        q_transposed = jnp.transpose(q_reshaped, (0, 2, 1, 3))  # (B, NH, S, DH)
        k_transposed = jnp.transpose(k_reshaped, (0, 2, 1, 3))  # (B, NH, S, DH)
        v_transposed = jnp.transpose(v_reshaped, (0, 2, 1, 3))  # (B, NH, S, DH)

        # Get causal mask of appropriate size
        causal_mask = self.causal_mask[:S, :S]

        # Apply mLSTM backend function
        h_state = self.backend_fn(
            queries=q_transposed,
            keys=k_transposed,
            values=v_transposed,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, S, DH)

        # Apply normalization
        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)

        # Reshape to original dimensions
        h_state_norm = jnp.transpose(h_state_norm, (0, 2, 1, 3))  # (B, S, NH, DH)
        h_state_norm = h_state_norm.reshape(B, S, -1)  # (B, S, H)

        return h_state_norm
