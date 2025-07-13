# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
import functools
import math
import typing as tp

import chex
import jax
import jax.numpy as jnp
from jax import lax


@functools.partial(
    jax.jit,
    static_argnames=("stabilize_rowwise", "eps"),
)
def parallel_stabilized_simple(
    queries: jax.Array,  # (B, NH, S, DH)
    keys: jax.Array,  # (B, NH, S, DH)
    values: jax.Array,  # (B, NH, S, DH)
    igate_preact: jax.Array,  # (B, NH, S, 1)
    fgate_preact: jax.Array,  # (B, NH, S, 1)
    lower_triangular_matrix: tp.Optional[jax.Array] = None,  # (S, S)
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
) -> jax.Array:
    """This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        queries: (B, NH, S, DH)
        keys: (B, NH, S, DH)
        values: (B, NH, S, DH)
        igate_preact: (B, NH, S, 1)
        fgate_preact: (B, NH, S, 1)
        lower_triangular_matrix: (S, S). Defaults to None.
        stabilize_rowwise: Whether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.
        eps: Small constant for numerical stability. Defaults to 1e-6.

    Returns:
        h_tilde_state: (B, NH, S, DH)
    """
    B, NH, S, DH = queries.shape
    _dtype = queries.dtype

    chex.assert_trees_all_equal_shapes(queries, keys, values)
    chex.assert_shape(igate_preact, (B, NH, S, 1))
    chex.assert_shape(fgate_preact, (B, NH, S, 1))

    # forget gate matrix
    log_fgates = jax.nn.log_sigmoid(fgate_preact)  # (B, NH, S, 1)
    ltr = lax.cond(
        lower_triangular_matrix is None or S != lower_triangular_matrix.shape[0],
        lambda: jnp.tril(jnp.ones((S, S), dtype=jnp.bool)),
        lambda: lower_triangular_matrix,
    )

    # Calculate cumulative sum of log forget gates
    log_fgates_cumsum = jnp.concatenate(
        [
            jnp.zeros((B, NH, 1, 1), dtype=_dtype),
            jnp.cumsum(log_fgates, axis=2),
        ],
        axis=2,
    )  # (B, NH, S+1, 1)

    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    # (B, NH, S+1, S+1)
    rep_log_fgates_cumsum = jnp.repeat(log_fgates_cumsum, S + 1, axis=-1)

    # now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    # (B, NH, S+1, S+1)
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.swapaxes(-2, -1)

    # causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = jnp.where(
        ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf")
    )  # (B, NH, S, S)

    # Gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + jnp.transpose(
        igate_preact, (0, 1, 3, 2)
    )  # (B, NH, S, S)

    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D = jnp.max(log_D_matrix, axis=-1, keepdims=True)  # (B, NH, S, 1)
    else:
        max_log_D = jnp.max(
            log_D_matrix.reshape(B, NH, -1), axis=-1, keepdims=True
        ).reshape(B, NH, 1, 1)  # (B, NH, 1, 1)

    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = jnp.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    # Scale keys by sqrt(DH)
    keys_scaled = keys / math.sqrt(DH)

    # Combination matrix C
    qk_matrix = queries @ keys_scaled.swapaxes(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)

    # Compute normalizer with numerical stability
    C_sum = jnp.sum(C_matrix, axis=-1, keepdims=True)
    normalizer = jnp.maximum(jnp.abs(C_sum), jnp.exp(-max_log_D))  # (B, NH, S, 1)

    # Normalize combination matrix
    C_matrix_normalized = C_matrix / (normalizer + eps)  # (B, NH, S, S)

    # Compute retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    return h_tilde_state


@functools.partial(jax.jit, static_argnames=("eps",))
def recurrent_step_stabilized_simple(
    c_state: jax.Array,  # (B, NH, DH, DH)
    n_state: jax.Array,  # (B, NH, DH, 1)
    m_state: jax.Array,  # (B, NH, 1, 1)
    q: jax.Array,  # (B, NH, 1, DH)
    k: jax.Array,  # (B, NH, 1, DH)
    v: jax.Array,  # (B, NH, 1, DH)
    igate_preact: jax.Array,  # (B, NH, 1, 1)
    fgate_preact: jax.Array,  # (B, NH, 1, 1)
    eps: float = 1e-6,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """This is a single step of the mLSTM operation in recurrent form.

    Args:
        c_state: (B, NH, DH, DH)
        n_state: (B, NH, DH, 1)
        m_state: (B, NH, 1, 1)
        q: (B, NH, 1, DH)
        k: (B, NH, 1, DH)
        v: (B, NH, 1, DH)
        igate_preact: (B, NH, 1, 1)
        fgate_preact: (B, NH, 1, 1)
        eps: Small constant for numerical stability. Defaults to 1e-6.

    Returns:
        tuple containing:
            - hidden_state: (B, NH, 1, DH)
            - tuple of new states: (c_state_new, n_state_new, m_state_new)
    """
    B, NH, S, DH = q.shape

    # Reshape inputs to ensure they have the right dimensions
    # Unlike PyTorch's in-place operations, we create new arrays
    q_reshaped = q.reshape(B, NH, DH, 1)  # (B, NH, DH, 1)
    k_reshaped = k.reshape(B, NH, DH, 1)  # (B, NH, DH, 1)
    v_reshaped = v.reshape(B, NH, DH, 1)  # (B, NH, DH, 1)

    # Compute gates
    log_fg_act = jax.nn.log_sigmoid(fgate_preact)  # (B, NH, 1, 1)

    # Update state with stabilization
    m_state_new = jnp.maximum(log_fg_act + m_state, igate_preact)  # (B, NH, 1, 1)

    # Compute activations
    fg_act = jnp.exp(log_fg_act + m_state - m_state_new)  # (B, NH, 1, 1)
    ig_act = jnp.exp(igate_preact - m_state_new)  # (B, NH, 1, 1)

    # Scale keys by sqrt(DH)
    k_scaled = k_reshaped / math.sqrt(DH)

    # Update context state
    c_state_new = fg_act * c_state + ig_act * (
        k_scaled @ jnp.transpose(v_reshaped, (0, 1, 3, 2))
    )  # (B, NH, DH, DH)

    # Update norm state
    n_state_new = fg_act * n_state + ig_act * k_scaled  # (B, NH, DH, 1)

    # Compute hidden state with numerical stabilization
    h_num = jnp.transpose(q_reshaped, (0, 1, 3, 2)) @ c_state_new  # (B, NH, 1, DH)

    qn_dotproduct = (
        jnp.transpose(q_reshaped, (0, 1, 3, 2)) @ n_state_new
    )  # (B, NH, 1, 1)
    max_val = jnp.exp(-m_state_new)  # (B, NH, 1, 1)
    h_denom = jnp.maximum(jnp.abs(qn_dotproduct), max_val) + eps
    h = h_num / h_denom  # (B, NH, 1, DH) / (B, NH, 1, 1) = (B, NH, 1, DH)

    return h, (c_state_new, n_state_new, m_state_new)


@functools.partial(
    jax.jit,
    static_argnames=("chunk_size", "return_last_state", "eps"),
)
def chunkwise_simple(
    queries: jax.Array,  # (B, NH, S, DH)
    keys: jax.Array,  # (B, NH, S, DH)
    values: jax.Array,  # (B, NH, S, DH)
    igate_preact: jax.Array,  # (B, NH, S)
    fgate_preact: jax.Array,  # (B, NH, S)
    initial_C: tp.Optional[jax.Array] = None,  # (B, NH, DH, DH)
    initial_n: tp.Optional[jax.Array] = None,  # (B, NH, DH)
    initial_m: tp.Optional[jax.Array] = None,  # (B, NH, 1)
    chunk_size: int = 64,
    return_last_state: bool = False,
    eps: float = 1e-6,
) -> tp.Union[jax.Array, tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]]:
    """Chunkwise mLSTM processing for efficient computation.

    Args:
        queries: (B, NH, S, DH)
        keys: (B, NH, S, DH)
        values: (B, NH, S, DH)
        igate_preact: (B, NH, S)
        fgate_preact: (B, NH, S)
        initial_C: Initial context state. Defaults to None.
        initial_n: Initial norm state. Defaults to None.
        initial_m: Initial max state. Defaults to None.
        chunk_size: Size of chunks to process. Defaults to 64.
        return_last_state: Whether to return the last state. Defaults to False.
        eps: Small constant for numerical stability. Defaults to 1e-6.

    Returns:
        Either output tensor or tuple of output tensor and final states
    """
    B, NH, S, DH = queries.shape
    NS, CS = S // chunk_size, chunk_size
    dtype = queries.dtype

    # Form chunks
    q = queries.reshape(B, NH, NS, CS, DH) / math.sqrt(DH)
    k = keys.reshape(B, NH, NS, CS, DH)
    v = values.reshape(B, NH, NS, CS, DH)

    # Compute forget gates
    log_fgates = jax.nn.log_sigmoid(fgate_preact).reshape(B, NH, NS, CS)
    log_fgates_acc = jnp.cumsum(log_fgates, axis=3)
    igate_preact_reshaped = igate_preact.reshape(B, NH, NS, CS)

    # Compute local gates
    loggates = (igate_preact_reshaped - log_fgates_acc)[:, :, :, :, None]

    # Compute max for stabilization (along chunk dimension)
    m_loc = jnp.max(
        loggates + log_fgates_acc[:, :, :, -1, None, None], axis=3, keepdims=True
    )

    # Stabilize loggates
    loggates = loggates + log_fgates_acc[:, :, :, -1, None, None] - m_loc

    # Compute key-value products
    kv = jnp.transpose(k, (0, 1, 2, 4, 3)) @ (v * jnp.exp(loggates))
    ksum = jnp.sum(k * jnp.exp(loggates), axis=-2)

    # Initialize states
    C = jnp.zeros((B, NH, NS + 1, DH, DH), dtype=dtype)
    n = jnp.zeros((B, NH, NS + 1, DH), dtype=dtype)
    m = jnp.zeros((B, NH, NS + 1, 1, 1), dtype=dtype)

    # Set initial states if provided
    if initial_C is not None:
        C = C.at[:, :, 0].set(initial_C)
    if initial_n is not None:
        n = n.at[:, :, 0].set(initial_n)
    if initial_m is not None:
        m = m.at[:, :, 0].set(initial_m[:, :, None, None])

    # Process chunks sequentially using a scan function
    def process_chunk(carry, idx):
        C_prev, n_prev, m_prev = carry
        i = idx + 1  # 1-based indexing since we initialized at position 0

        # Compute new m state
        m_new = jnp.maximum(
            log_fgates_acc[:, :, i - 1, -1, None, None] + m_prev, m_loc[:, :, i - 1]
        )

        # Compute new C state
        C_new = C_prev * jnp.exp(
            log_fgates_acc[:, :, i - 1, -1, None, None] + m_prev - m_new
        ) + kv[:, :, i - 1] * jnp.exp(m_loc[:, :, i - 1] - m_new)

        # Compute new n state
        n_new = n_prev * jnp.exp(
            log_fgates_acc[:, :, i - 1, -1, None]
            + m_prev[:, :, :, 0]
            - m_new[:, :, :, 0]
        ) + ksum[:, :, i - 1] * jnp.exp(m_loc[:, :, i - 1, 0] - m_new[:, :, :, 0])

        # Store states in the arrays
        C_out = C.at[:, :, i].set(C_new)
        n_out = n.at[:, :, i].set(n_new)
        m_out = m.at[:, :, i].set(m_new)

        return (C_new, n_new, m_new), (C_out, n_out, m_out)

    # Use scan for more efficient jit compilation
    _, (C_final, n_final, m_final) = jax.lax.scan(
        process_chunk, (C[:, :, 0], n[:, :, 0], m[:, :, 0]), jnp.arange(NS)
    )

    # Create causal mask for chunk processing
    log_fgates_rep = jnp.repeat(log_fgates_acc[:, :, :, :, None], CS, axis=4)

    triu_mask = jnp.triu(jnp.ones((CS, CS), dtype=bool), k=1)

    log_fg_matrix = log_fgates_rep - jnp.transpose(log_fgates_rep, (0, 1, 2, 4, 3))
    log_fg_matrix = jnp.where(triu_mask, -float("inf"), log_fg_matrix)

    # Gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + jnp.transpose(
        igate_preact_reshaped[:, :, :, :, None], (0, 1, 2, 4, 3)
    )  # (B, NH, NS, CS, CS)

    D_max = jnp.max(log_D_matrix, axis=-1, keepdims=True)

    # Compute stabilization factor
    stab = jnp.maximum(D_max, m_final[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None])

    # Compute inter-chunk attention
    inter_C = (
        q * jnp.exp(m_final[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None] - stab)
    ) @ C_final[:, :, :-1]

    inter_n = (
        q * jnp.exp(m_final[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None] - stab)
    ) @ n_final[:, :, :-1, :, None]

    # Stabilize D matrix
    log_D_matrix_stabilized = log_D_matrix - stab
    D_matrix = jnp.exp(log_D_matrix_stabilized)

    # Compute combination matrix C
    qk_matrix = q @ jnp.transpose(k, (0, 1, 2, 4, 3))
    E_matrix = qk_matrix * D_matrix

    # Compute normalizer with numerical stability
    normalizer = jnp.maximum(
        jnp.abs(jnp.sum(E_matrix, axis=-1, keepdims=True) + inter_n), jnp.exp(-stab)
    )

    # Normalize and compute attention outputs
    E_matrix_normalized = E_matrix / (normalizer + eps)
    intra = E_matrix_normalized @ v
    inter = inter_C / (normalizer + eps)

    # Combine intra- and inter-chunk outputs
    output = (intra + inter).reshape(B, NH, S, DH)

    return lax.cond(
        return_last_state,
        lambda _: (output, (C_final[:, :, -1], n_final[:, :, -1], m_final[:, :, -1])),
        lambda _: (output, (None, None, None)),
        operand=None,
    )

    # if return_last_state:
    #     return output, (C_final[:, :, -1], n_final[:, :, -1], m_final[:, :, -1])
    # else:
    #     return output
