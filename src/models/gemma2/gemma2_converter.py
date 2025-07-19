"""
Utility for converting Hugging Face Gemma2 model weights to Flax implementation.
"""

import typing as tp
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import Mesh
from transformers import Gemma2Config, Gemma2ForCausalLM

from .modeling_gemma2 import Gemma2ForCausalLM as FlaxGemma2ForCausalLM


class Gemma2WeightConverter:
    """
    Converts Hugging Face Gemma2 model weights to Flax NNX format.

    This class handles the conversion of PyTorch tensors to JAX arrays and
    maps the weight names between the two model implementations.
    """

    def __init__(self, config: Gemma2Config):
        self.config = config
        self._weight_mapping = self._create_weight_mapping()

    def _create_weight_mapping(self) -> dict:
        """Create mapping from HF weight names to Flax weight names."""
        mapping = {}

        # Embedding layer
        mapping["model.embed_tokens.weight"] = "model.embed_tokens.embedding"

        # Final layer norm
        mapping["model.norm.weight"] = "model.norm.scale"

        # Language modeling head
        mapping["lm_head.weight"] = "lm_head.kernel"

        # Decoder layers
        for i in range(self.config.num_hidden_layers):
            layer_prefix = f"model.layers.{i}"
            flax_layer_prefix = f"model.layers.{i}"

            # Attention projections
            mapping[f"{layer_prefix}.self_attn.q_proj.weight"] = (
                f"{flax_layer_prefix}.self_attn.q_proj.kernel"
            )
            mapping[f"{layer_prefix}.self_attn.k_proj.weight"] = (
                f"{flax_layer_prefix}.self_attn.k_proj.kernel"
            )
            mapping[f"{layer_prefix}.self_attn.v_proj.weight"] = (
                f"{flax_layer_prefix}.self_attn.v_proj.kernel"
            )
            mapping[f"{layer_prefix}.self_attn.o_proj.weight"] = (
                f"{flax_layer_prefix}.self_attn.o_proj.kernel"
            )

            # Attention biases (if present)
            if self.config.attention_bias:
                mapping[f"{layer_prefix}.self_attn.q_proj.bias"] = (
                    f"{flax_layer_prefix}.self_attn.q_proj.bias"
                )
                mapping[f"{layer_prefix}.self_attn.k_proj.bias"] = (
                    f"{flax_layer_prefix}.self_attn.k_proj.bias"
                )
                mapping[f"{layer_prefix}.self_attn.v_proj.bias"] = (
                    f"{flax_layer_prefix}.self_attn.v_proj.bias"
                )
                mapping[f"{layer_prefix}.self_attn.o_proj.bias"] = (
                    f"{flax_layer_prefix}.self_attn.o_proj.bias"
                )

            # MLP projections
            mapping[f"{layer_prefix}.mlp.gate_proj.weight"] = (
                f"{flax_layer_prefix}.mlp.gate_proj.kernel"
            )
            mapping[f"{layer_prefix}.mlp.up_proj.weight"] = (
                f"{flax_layer_prefix}.mlp.up_proj.kernel"
            )
            mapping[f"{layer_prefix}.mlp.down_proj.weight"] = (
                f"{flax_layer_prefix}.mlp.down_proj.kernel"
            )

            # Layer norms
            mapping[f"{layer_prefix}.input_layernorm.weight"] = (
                f"{flax_layer_prefix}.input_layernorm.scale"
            )
            mapping[f"{layer_prefix}.post_attention_layernorm.weight"] = (
                f"{flax_layer_prefix}.post_attention_layernorm.scale"
            )
            mapping[f"{layer_prefix}.pre_feedforward_layernorm.weight"] = (
                f"{flax_layer_prefix}.pre_feedforward_layernorm.scale"
            )
            mapping[f"{layer_prefix}.post_feedforward_layernorm.weight"] = (
                f"{flax_layer_prefix}.post_feedforward_layernorm.scale"
            )

        return mapping

    def _torch_to_jax(self, tensor: torch.Tensor) -> jax.Array:
        """Convert PyTorch tensor to JAX array."""
        return jnp.array(tensor.detach().cpu().numpy())

    def _transpose_linear_weight(self, weight: torch.Tensor) -> jax.Array:
        """
        Transpose linear layer weights for Flax format.
        PyTorch: [out_features, in_features]
        Flax: [in_features, out_features]
        """
        return jnp.transpose(self._torch_to_jax(weight))

    def _get_nested_attr(self, obj, path: str):
        """Get nested attribute from object using dot notation."""
        attrs = path.split(".")
        for attr in attrs:
            if attr.isdigit():
                obj = obj[int(attr)]
            else:
                obj = getattr(obj, attr)
        return obj

    def _set_nested_attr(self, obj, path: str, value):
        """Set nested attribute on object using dot notation."""
        attrs = path.split(".")
        for attr in attrs[:-1]:
            if attr.isdigit():
                obj = obj[int(attr)]
            else:
                obj = getattr(obj, attr)

        final_attr = attrs[-1]
        if final_attr.isdigit():
            obj[int(final_attr)] = value
        else:
            setattr(obj, final_attr, value)

    def convert_weights(
        self,
        hf_model: Gemma2ForCausalLM,
        flax_model: FlaxGemma2ForCausalLM,
    ) -> FlaxGemma2ForCausalLM:
        """
        Convert weights from HuggingFace model to Flax model.

        Args:
            hf_model: Loaded HuggingFace Gemma2 model
            flax_model: Initialized Flax Gemma2 model

        Returns:
            Flax model with converted weights
        """
        hf_state_dict = hf_model.state_dict()

        # Convert weights
        for hf_name, flax_path in self._weight_mapping.items():
            if hf_name not in hf_state_dict:
                print(f"Warning: {hf_name} not found in HuggingFace model")
                continue

            hf_weight = hf_state_dict[hf_name]

            # Handle weight transformations
            if "kernel" in flax_path and "embed" not in flax_path:
                # Linear layer weights need transposition
                jax_weight = self._transpose_linear_weight(hf_weight)
            elif "scale" in flax_path:
                # RMSNorm weights (keep as is)
                jax_weight = self._torch_to_jax(hf_weight)
            elif "embedding" in flax_path:
                # Embedding weights (keep as is)
                jax_weight = self._torch_to_jax(hf_weight)
            elif "bias" in flax_path:
                # Bias terms (keep as is)
                jax_weight = self._torch_to_jax(hf_weight)
            else:
                # Default: keep as is
                jax_weight = self._torch_to_jax(hf_weight)

            # Set the weight in the Flax model
            try:
                self._set_nested_attr(flax_model, flax_path, jax_weight)
                print(f"Converted: {hf_name} -> {flax_path}")
            except Exception as e:
                print(f"Error converting {hf_name} -> {flax_path}: {e}")

        return flax_model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> FlaxGemma2ForCausalLM:
        """
        Load a pretrained HuggingFace Gemma2 model and convert to Flax.

        Args:
            model_name_or_path: HuggingFace model name or path
            mesh: JAX mesh for sharding
            rngs: Random number generators
            dtype: Model dtype

        Returns:
            Flax Gemma2 model with converted weights
        """
        # Load HuggingFace model
        print(f"Loading HuggingFace model: {model_name_or_path}")
        hf_model = Gemma2ForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        # Get config
        config = hf_model.config

        # Initialize Flax model
        print("Initializing Flax model...")
        flax_model = FlaxGemma2ForCausalLM(
            config=config,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
        )

        # Convert weights
        print("Converting weights...")
        converter = cls(config)
        converted_model = converter.convert_weights(hf_model, flax_model)

        print("Conversion complete!")
        return converted_model

    def save_converted_model(
        self,
        model: FlaxGemma2ForCausalLM,
        save_path: tp.Union[str, Path],
    ):
        """
        Save the converted Flax model.

        Args:
            model: Converted Flax model
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        state = nnx.state(model)

        # Convert to serializable format
        state_dict = {}
        for key, value in state.items():
            if isinstance(value, jax.Array):
                state_dict[key] = np.array(value)
            else:
                state_dict[key] = value

        # Save using numpy
        np.savez_compressed(save_path / "flax_model.npz", **state_dict)

        # Save config
        model.config.save_pretrained(save_path)

        print(f"Model saved to {save_path}")
