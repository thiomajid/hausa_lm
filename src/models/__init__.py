"""
JAX/Flax model implementations.

This module contains JAX implementations of transformer models using Flax NNX.
"""

import typing as tp

from flax import nnx

from .gemma2.modeling_gemma2 import Gemma2ForCausalLM

MODEL_REGISTRY: dict[str, tp.Type[nnx.Module]] = {"gemma2": Gemma2ForCausalLM}

__all__ = [
    "MODEL_REGISTRY",
    "Gemma2ForCausalLM",
]
