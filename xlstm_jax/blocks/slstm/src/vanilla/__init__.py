# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbininan Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano

import functools
from typing import Callable

import chex
import jax
import jax.numpy as jnp

from .lstm import slstm_forward_pointwise as slstm_forward_pointwise_lstm
from .slstm import slstm_forward_pointwise as slstm_forward_pointwise_slstm

slstm_pointwise_function_registry: dict[str, Callable] = {
    "slstm": slstm_forward_pointwise_slstm,
    "lstm": slstm_forward_pointwise_lstm,
}


@functools.partial(
    jax.jit,
    static_argnames=("pointwise_forward", "mesh"),
)
def slstm_forward(
    x: jax.Array,  # [S, B, G*I]
    states: jax.Array,  # [4, B, H] only the first is used for recurrence!
    R: jax.Array,  # [K, R*H, H] - K num_heads
    b: jax.Array,  # [T*H]
    pointwise_forward: Callable[
        [jax.Array, jax.Array, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array],
    ],
    mesh: jax.sharding.Mesh,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Forward pass for sLSTM over a full sequence.

    Args:
        x: Input tensor of shape [S, B, G*I] (sequence, batch, features)
        states: Initial states tensor of shape [4, B, H] (4: num_states, B: batch, H: hidden_dim)
        R: Recurrent kernel of shape [K, R*H, H] (K: num_heads, R: num_gates, H: head_dim)
        b: Recurrent bias of shape [T*H]
        pointwise_forward: Function implementing the pointwise operations

    Returns:
        tuple of:
        - All states for the sequence: [num_states, S+1, B, H]
        - Final state: [num_states, B, H]
        - Gate activations: [S+1, num_gates, B, H]
    """

    num_states = states.shape[0]
    sequence_dim = x.shape[0]
    num_gates_r = R.shape[1] // R.shape[2]  # Only works for fully-connected RNN
    hidden_dim = R.shape[2] * R.shape[0]
    num_gates_t = b.shape[0] // hidden_dim
    batch_dim = x.shape[1]
    num_heads = R.shape[0]
    head_dim = R.shape[2]

    chex.assert_equal(batch_dim, states.shape[1])
    chex.assert_equal(hidden_dim, states.shape[2])

    g = jnp.zeros(
        shape=(sequence_dim + 1, num_gates_t, batch_dim, hidden_dim),
        dtype=x.dtype,
    )

    states_all = jnp.zeros(
        (num_states, sequence_dim + 1, batch_dim, hidden_dim), dtype=x.dtype
    )

    states_all = states_all.at[:, 0].set(states)

    def process_timestep(carry: tuple[jax.Array, ...], time_idx):
        current_states, states_all, g = carry
        current_x = x[time_idx]

        hidden_reshaped = current_states[0].reshape(batch_dim, num_heads, 1, head_dim)
        R_reshaped = jnp.transpose(R, (0, 2, 1)).reshape(
            1, num_heads, head_dim, num_gates_r * head_dim
        )
        Ry = jnp.matmul(hidden_reshaped, R_reshaped).reshape(
            batch_dim, num_heads, num_gates_r, head_dim
        )

        Ry = jnp.transpose(Ry, (0, 2, 1, 3)).reshape(batch_dim, -1)

        new_states, gates = pointwise_forward(
            current_x,
            Ry,
            b,
            current_states,
            mesh=mesh,
        )

        new_states = new_states.astype(current_states.dtype)
        gates = gates.astype(current_states.dtype)

        states_all = states_all.at[:, time_idx + 1].set(new_states)
        g = g.at[time_idx].set(gates)

        return (new_states, states_all, g), None

    init_carry = (states, states_all, g)
    (final_state, states_all, g), _ = jax.lax.scan(
        f=process_timestep,
        init=init_carry,
        xs=jnp.arange(sequence_dim),
    )

    return states_all, final_state, g


@functools.partial(
    jax.jit,
    static_argnames=("pointwise_forward",),
)
def slstm_forward_step(
    x: jax.Array,  # [B, G*I]
    states: jax.Array,  # [4, B, H] only the first is used for recurrence!
    R: jax.Array,  # [K, R*H, H] - K num_heads
    b: jax.Array,  # [T*H]
    pointwise_forward: Callable,
) -> tuple[jax.Array, jax.Array]:
    """
    Forward pass for a single step of the sLSTM.

    Args:
        x: Input tensor of shape [B, G*I] (batch, features)
        states: Current states tensor of shape [4, B, H]
        R: Recurrent weights tensor of shape [K, R*H, H] where K is num_heads
        b: Bias tensor of shape [T*H]
        pointwise_forward: Function implementing the pointwise operations

    Returns:
        tuple of:
        - New state: [num_states, 1, B, H]
        - Gate activations: [1, num_gates, B, H]
    """
    # num_states = states.shape[0]
    batch_dim = states.shape[1]
    # hidden_dim = states.shape[2]
    num_heads = R.shape[0]
    head_dim = R.shape[2]
    num_gates_r = R.shape[1] // R.shape[2]
    # num_gates_t = b.shape[0] // hidden_dim

    # Initialize arrays
    # g = jnp.zeros(
    #     (1, num_gates_t, batch_dim, hidden_dim),
    #     dtype=x.dtype,
    # )

    # Compute recurrent projection
    # Reshape hidden state to [B, NH, 1, HD]
    hidden_reshaped = states[0].reshape(batch_dim, num_heads, 1, head_dim)

    # Reshape and transpose R to [1, NH, HD, NG*HD]
    R_reshaped = jnp.transpose(R, (0, 2, 1)).reshape(
        1, num_heads, head_dim, num_gates_r * head_dim
    )

    # Matrix multiply and reshape to [B, -1]
    Ry = jnp.matmul(hidden_reshaped, R_reshaped).reshape(
        batch_dim, num_heads, num_gates_r, head_dim
    )
    Ry = jnp.transpose(Ry, (0, 2, 1, 3)).reshape(batch_dim, -1)

    # Process input with slstm
    x_input = x
    if x.ndim == 3:  # Handle optional sequence dimension
        x_input = x[0]

    new_states, gates = pointwise_forward(x_input, Ry, b, states)

    # Add sequence dimension if needed
    new_states_with_seq = new_states[:, None, ...]
    new_gates_with_seq = gates[None, ...]

    return new_states_with_seq, new_gates_with_seq
