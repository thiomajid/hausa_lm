"""
Utility for converting Hugging Face Qwen3 model weights to Flax implementation.

This module provides functionality to convert PyTorch-based Qwen3 models from
Hugging Face to JAX/Flax NNX format, handling weight mapping and tensor conversion.
"""

import typing as tp
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Conditional imports for PyTorch/HuggingFace dependencies
try:
    import torch
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3ForCausalLM as HFQwen3ForCausalLM,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Use our local config if HF is not available
    from . import Qwen3Config

from . import Qwen3ForCausalLM as FlaxQwen3ForCausalLM


class Qwen3WeightConverter:
    """
    Converts Hugging Face Qwen3 model weights to Flax NNX format.

    This class handles the conversion of PyTorch tensors to JAX arrays and
    maps the weight names between the two model implementations.
    """

    def __init__(self, config: Qwen3Config):
        self.config = config
        self._weight_mapping = self._create_weight_mapping()

    def _create_weight_mapping(self) -> dict:
        """Create mapping from HF weight names to Flax weight names."""
        mapping = {}

        # Embedding layer
        mapping["model.embed_tokens.weight"] = "model.embed_tokens.embedding"

        # Final layer norm
        mapping["model.norm.weight"] = "model.norm.weight"

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

            # Query/Key normalization (unique to Qwen3)
            mapping[f"{layer_prefix}.self_attn.q_norm.weight"] = (
                f"{flax_layer_prefix}.self_attn.q_norm.weight"
            )
            mapping[f"{layer_prefix}.self_attn.k_norm.weight"] = (
                f"{flax_layer_prefix}.self_attn.k_norm.weight"
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

            # MLP projections (SwiGLU: gate, up, down)
            mapping[f"{layer_prefix}.mlp.gate_proj.weight"] = (
                f"{flax_layer_prefix}.mlp.gate_proj.kernel"
            )
            mapping[f"{layer_prefix}.mlp.up_proj.weight"] = (
                f"{flax_layer_prefix}.mlp.up_proj.kernel"
            )
            mapping[f"{layer_prefix}.mlp.down_proj.weight"] = (
                f"{flax_layer_prefix}.mlp.down_proj.kernel"
            )

            # Layer normalizations
            mapping[f"{layer_prefix}.input_layernorm.weight"] = (
                f"{flax_layer_prefix}.input_layernorm.weight"
            )
            mapping[f"{layer_prefix}.post_attention_layernorm.weight"] = (
                f"{flax_layer_prefix}.post_attention_layernorm.weight"
            )

        return mapping

    def _torch_to_jax(self, tensor: "torch.Tensor") -> jax.Array:
        """Convert PyTorch tensor to JAX array."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for tensor conversion")
        return jnp.array(tensor.detach().cpu().numpy())

    def _transpose_linear_weight(self, weight: "torch.Tensor") -> jax.Array:
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
            # For nnx.Param objects, we need to set the .value attribute
            param_obj = getattr(obj, final_attr)
            if hasattr(param_obj, "value"):
                param_obj.value = value
            else:
                setattr(obj, final_attr, value)

    def convert_weights(
        self,
        hf_model: "HFQwen3ForCausalLM",
        flax_model: FlaxQwen3ForCausalLM,
    ) -> FlaxQwen3ForCausalLM:
        """
        Convert weights from HuggingFace model to Flax model.

        Args:
            hf_model: Loaded HuggingFace Qwen3 model
            flax_model: Initialized Flax Qwen3 model

        Returns:
            Flax model with converted weights
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and transformers required for weight conversion")

        hf_state_dict = hf_model.state_dict()
        print(f"Converting {len(hf_state_dict)} weight tensors...")

        # Convert weights
        converted_count = 0
        for hf_name, flax_path in self._weight_mapping.items():
            if hf_name not in hf_state_dict:
                print(f"Warning: {hf_name} not found in HuggingFace model")
                continue

            hf_weight = hf_state_dict[hf_name]

            # Handle weight transformations based on layer type
            if "kernel" in flax_path and "embed" not in flax_path:
                # Linear layer weights need transposition
                jax_weight = self._transpose_linear_weight(hf_weight)
            elif "weight" in flax_path and (
                "norm" in flax_path or "layernorm" in flax_path
            ):
                # RMS normalization weights (keep as is)
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
                converted_count += 1
                if converted_count % 10 == 0:
                    print(f"Converted {converted_count} tensors...")
            except Exception as e:
                print(f"Error converting {hf_name} -> {flax_path}: {e}")

        print(
            f"Successfully converted {converted_count}/{len(self._weight_mapping)} tensors"
        )
        return flax_model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        **hf_kwargs,
    ) -> FlaxQwen3ForCausalLM:
        """
        Load a pretrained HuggingFace Qwen3 model and convert to Flax.

        Args:
            model_name_or_path: HuggingFace model name or path
            rngs: Random number generators for Flax model initialization
            dtype: Model dtype
            **hf_kwargs: Additional arguments for HuggingFace model loading

        Returns:
            Flax Qwen3 model with converted weights
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and transformers required for loading pretrained models"
            )

        # Load HuggingFace model
        print(f"Loading HuggingFace model: {model_name_or_path}")
        hf_model = HFQwen3ForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float32, device_map="cpu", **hf_kwargs
        )

        # Get config
        config = hf_model.config

        # Initialize Flax model
        print("Initializing Flax model...")
        flax_model = FlaxQwen3ForCausalLM(config=config, rngs=rngs)

        # Convert weights
        print("Converting weights...")
        converter = cls(config)
        converted_model = converter.convert_weights(hf_model, flax_model)

        print("Conversion complete!")
        return converted_model

    def save_converted_model(
        self,
        model: FlaxQwen3ForCausalLM,
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

        # Save config if it has a save method
        if hasattr(model.config, "save_pretrained"):
            model.config.save_pretrained(save_path)
        else:
            # Save config as JSON
            import json

            config_dict = {
                attr: getattr(model.config, attr)
                for attr in dir(model.config)
                if not attr.startswith("_")
                and not callable(getattr(model.config, attr))
            }
            with open(save_path / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

        print(f"Model saved to {save_path}")

    @staticmethod
    def load_converted_model(
        load_path: tp.Union[str, Path],
        rngs: nnx.Rngs,
    ) -> FlaxQwen3ForCausalLM:
        """
        Load a previously converted and saved Flax model.

        Args:
            load_path: Path to the saved model
            rngs: Random number generators for model initialization

        Returns:
            Loaded Flax Qwen3 model
        """
        load_path = Path(load_path)

        # Load config
        config_path = load_path / "config.json"
        if config_path.exists():
            import json

            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = Qwen3Config(**config_dict)
        else:
            # Try HuggingFace config if available
            if TORCH_AVAILABLE:
                from transformers import Qwen3Config as HFQwen3Config

                config = HFQwen3Config.from_pretrained(load_path)
            else:
                raise FileNotFoundError(f"Config file not found at {config_path}")

        # Initialize model
        model = FlaxQwen3ForCausalLM(config=config, rngs=rngs)

        # Load weights
        state_path = load_path / "flax_model.npz"
        if state_path.exists():
            loaded_state = dict(np.load(state_path))

            # Convert back to JAX arrays
            jax_state = {}
            for key, value in loaded_state.items():
                jax_state[key] = jnp.array(value)

            # Update model state
            nnx.update(model, jax_state)
        else:
            raise FileNotFoundError(f"Model weights not found at {state_path}")

        print(f"Model loaded from {load_path}")
        return model


def convert_qwen3_from_hf(
    model_name: str,
    save_path: tp.Optional[tp.Union[str, Path]] = None,
    rngs: tp.Optional[nnx.Rngs] = None,
    **hf_kwargs,
) -> FlaxQwen3ForCausalLM:
    """
    Convenience function to convert a Qwen3 model from HuggingFace format.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")
        save_path: Optional path to save the converted model
        rngs: Random number generators (uses default if None)
        **hf_kwargs: Additional arguments for HuggingFace model loading

    Returns:
        Converted Flax Qwen3 model

    Example:
        >>> model = convert_qwen3_from_hf("Qwen/Qwen3-0.6B", save_path="./qwen3_flax")
        >>> outputs = model(input_ids=jnp.array([[1, 2, 3, 4]]))
    """
    if rngs is None:
        rngs = nnx.Rngs(0)

    # Convert model
    model = Qwen3WeightConverter.from_pretrained(model_name, rngs=rngs, **hf_kwargs)

    # Save if path provided
    if save_path is not None:
        converter = Qwen3WeightConverter(model.config)
        converter.save_converted_model(model, save_path)

    return model


# Example usage
if __name__ == "__main__":
    """
    Example script showing how to convert Qwen3 models.
    """
    if not TORCH_AVAILABLE:
        print("PyTorch and transformers not available. Cannot run conversion example.")
        exit(1)

    # Convert a model
    model_name = "Qwen/Qwen3-0.6B"  # or any other Qwen3 model
    save_path = "./converted_qwen3"

    print(f"Converting {model_name} to Flax format...")

    try:
        model = convert_qwen3_from_hf(
            model_name, save_path=save_path, rngs=nnx.Rngs(42)
        )

        # Test the converted model
        print("Testing converted model...")
        test_input = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
        outputs = model(test_input)
        print(f"Test output shape: {outputs.logits.shape}")

        # Load the saved model
        print("Testing model loading...")
        loaded_model = Qwen3WeightConverter.load_converted_model(
            save_path, rngs=nnx.Rngs(42)
        )

        # Verify loaded model works
        loaded_outputs = loaded_model(test_input)
        print(f"Loaded model output shape: {loaded_outputs.logits.shape}")

        # Check if outputs are similar
        diff = jnp.abs(outputs.logits - loaded_outputs.logits).max()
        print(f"Max difference between original and loaded model: {diff}")

        print("Conversion and testing completed successfully!")

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback

        traceback.print_exc()
