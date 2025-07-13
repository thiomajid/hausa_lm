# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbininan Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano

import jax
import jax.numpy as jnp


@jax.jit
def slstm_forward_pointwise(
    Wx: jax.Array,  # dim [B, 4*H]
    Ry: jax.Array,  # dim [B, 4*H]
    b: jax.Array,  # dim [1, 4*H]
    states: jax.Array,  # dim [4, B, H]
) -> tuple[
    jax.Array,
    jax.Array,
]:
    """
    Implements the vanilla LSTM forward pass operation in JAX.

    Args:
        Wx: Input projections with shape [B, 4*H]
        Ry: Recurrent projections with shape [B, 4*H]
        b: Bias terms with shape [1, 4*H]
        states: Previous states with shape [4, B, H]

    Returns:
        Tuple containing:
            - New states [2, B, H] (hidden and cell)
            - Gate activations [4, B, H] (input, forget, cell, output)
    """

    # Combine input projection, recurrent projection, and bias
    raw = Wx + Ry + b

    # Extract cell state from the states tensor (hidden is at index 0, cell at index 1)
    states_reshaped = states.reshape(2, states.shape[1], -1)
    c = states_reshaped[1]  # cell state

    # Split the raw activations into the 4 gates
    raw_reshaped = raw.reshape(raw.shape[0], 4, -1)
    iraw = raw_reshaped[:, 0]  # input gate
    fraw = raw_reshaped[:, 1]  # forget gate
    zraw = raw_reshaped[:, 2]  # cell update
    oraw = raw_reshaped[:, 3]  # output gate

    # Compute gate activations
    igate = jax.nn.sigmoid(iraw)
    fgate = jax.nn.sigmoid(fraw)
    zval = jnp.tanh(zraw)
    ogate = jax.nn.sigmoid(oraw)

    # Update cell state
    cnew = fgate * c + igate * zval

    # Compute new hidden state
    ynew = ogate * jnp.tanh(cnew)

    # Stack the new states and gate activations
    new_states = jnp.stack((ynew, cnew), axis=0)
    gate_activations = jnp.stack((igate, fgate, zraw, ogate), axis=0)

    return new_states, gate_activations
