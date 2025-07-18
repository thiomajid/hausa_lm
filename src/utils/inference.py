import functools

import jax
import jax.numpy as jnp
from flax import nnx

GenerationCarry = tuple[
    jax.Array,  # The full sequence array
    int,  # The current index in the sequence
    jax.Array,  # The current PRNG key
]


# Define the generation step function globally or inside the jitted function
# Making it standalone allows for cleaner separation
def _generation_step_body(
    model: nnx.Module,  # Pass the model object
    carry: GenerationCarry,
    _,  # Loop variable from scan, unused
    vocab_size: int,
    temperature: float,
    greedy: bool,  # This will be a static argument passed to lax.cond predicate
):
    """Body function for one step of lax.scan."""
    current_full_x, current_index, current_key = carry
    step_key, next_key = jax.random.split(current_key)  # Split key for this step

    # --- Input Preparation ---
    input_sequence = current_full_x
    out = model.generate(input_sequence)  # Shape: [batch, seq_len, vocab_size]
    batch_size = current_full_x.shape[0]  # Get batch size dynamically

    # --- Logit Selection ---
    logits_slice = jax.lax.dynamic_slice(
        out,
        start_indices=(0, current_index - 1, 0),  # Last token logits
        slice_sizes=(batch_size, 1, vocab_size),  # whole batch, 1 token, vocab size
    )

    # (B, 1, V) -> (B, V)
    last_token_logits = jnp.squeeze(logits_slice, axis=1)

    # --- Sampling ---
    # Define functions for true/false branches of lax.cond
    def _greedy_sample(logits, _, __):
        return jnp.argmax(logits, axis=-1)

    def _temperature_sample(logits, temp, key):
        scaled_logits = logits / temp
        probabilities = jax.nn.softmax(scaled_logits, axis=-1)
        return jax.random.categorical(key, probabilities, axis=-1)

    next_token = jax.lax.cond(
        greedy,
        _greedy_sample,
        lambda logits, temp, key: _temperature_sample(logits, temp, key),
        last_token_logits,
        temperature,
        step_key,
    )

    next_token = next_token.astype(jnp.int32)

    # --- State update ---
    # Update the full sequence array with the new token
    updated_full_x = jax.lax.dynamic_update_slice(
        current_full_x, next_token[:, None], (0, current_index)
    )

    # Return new carry and the collected token
    return (
        updated_full_x,
        current_index + 1,
        next_key,
    ), next_token


@functools.partial(
    nnx.jit,
    static_argnames=("max_new_tokens", "vocab_size", "temperature", "greedy"),
)
def generate_sequence_scan(
    model: nnx.Module,
    initial_carry_val: GenerationCarry,
    max_new_tokens: int,
    vocab_size: int,
    temperature: float = 1.0,
    greedy: bool = False,
):
    """
    Generates a sequence using jax.lax.scan with support for greedy or temperature sampling.
    This function is JIT-compiled.

    Args:
        model: The NNX model object.
        initial_carry_val: Tuple containing (initial_sequence, initial_index, initial_prng_key).
        max_new_tokens: The number of new tokens to generate (static for JIT).
        vocab_size: The size of the vocabulary (static for JIT).
        temperature: Temperature for sampling (static for JIT).
        greedy: If True, use greedy decoding (static for JIT).

    Returns:
        The final generated sequence array.
    """
    # Create a generation step function specific to this call's static arguments
    generation_step_partial = functools.partial(
        _generation_step_body,
        model,
        vocab_size=vocab_size,
        temperature=temperature,
        greedy=greedy,
    )

    final_carry, _ = jax.lax.scan(
        generation_step_partial,  # Use the partial function
        initial_carry_val,
        None,  # xs, not needed here as we iterate based on length
        length=max_new_tokens,
    )
    final_x = final_carry[0]  # The full sequence array
    return final_x
