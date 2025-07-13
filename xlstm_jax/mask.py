import jax
import jax.numpy as jnp
from jax import lax


def create_padding_mask(input_ids: jax.Array, pad_token_idx: int):
    """
    Creates a binary mask used to zero-out padding tokens

    Args:
        input_ids (jax.Array): The input sequence(s) of shape [B, S]
        pad_token_idx (int): ID of the padding token in the model's vocabulary
    """
    is_padding = lax.eq(input_ids, pad_token_idx).astype(input_ids.dtype)
    mask = 1.0 - is_padding
    return mask


def apply_padding_mask_with_gradient_stop(embeddings: jax.Array, mask: jax.Array):
    """
    Apply padding mask while preventing gradients from flowing to padding positions.

    This is equivalent to PyTorch's padding_idx behavior in nn.Embedding.

    Args:
        embeddings: Word embeddings of shape [B, S, H]
        mask: Padding mask of shape [B, S] where 1.0 = valid, 0.0 = padding

    Returns:
        Masked embeddings with gradients stopped for padding positions
    """
    # Expand mask to match embedding dimensions [B, S, H]
    mask_expanded = mask[..., None]  # [B, S, 1]

    # For padding positions, use stop_gradient to prevent gradient flow
    # For valid positions, allow gradients to flow normally
    masked_embeddings = jnp.where(
        mask_expanded,
        embeddings,  # Valid positions: gradients flow
        lax.stop_gradient(embeddings),  # Padding positions: stop gradients
    )

    # Zero out the padding positions
    return masked_embeddings * mask_expanded
