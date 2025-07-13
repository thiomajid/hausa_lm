# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbininan Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from typing import Callable

import jax
from flax import nnx


def round_to_multiple(n, m=8):
    return ((n + m - 1) // m) * m


def conditional_decorator(condition, decorator):
    """A higher-order decorator that applies 'decorator' only if 'condition' is True."""

    def dummy_decorator(func):
        """A dummy decorator that does nothing."""
        return func

    if condition:
        # If condition is True, return the actual decorator
        return decorator
    else:
        # If condition is False, return the dummy decorator
        return dummy_decorator


class ParameterProxy:
    """
    This class helps keeping parameters in a specialized internal structure to be optimal for
    computation speed, while having a proxied version to be called externally that is backend-agnostic.

    It takes a module and a parameter name of a parameter in that module it represents.
    Via __setitem__ and __getitem__ the "external" version of the parameter can be accessed.

    Adapted for JAX/Flax's immutable nature where parameters are updated by returning new values.
    """

    def __init__(
        self,
        module: nnx.Module,
        parameter_name: str,
        internal_to_external: Callable[[jax.Array], jax.Array],
        external_to_internal: Callable[[jax.Array], jax.Array],
    ):
        self.module = module
        self.parameter_name = parameter_name
        self.internal_to_external = internal_to_external
        self.external_to_internal = external_to_internal

    def __getitem__(self, key):
        # Transform and then apply the slice to the external shape
        external_param = self.internal_to_external(
            getattr(self.module, self.parameter_name)
        )
        return external_param[key]

    def __setitem__(self, key, value):
        # In JAX/Flax, we handle parameter updates differently due to immutability
        # This method will modify the module's parameter in-place (for nnx this is allowed)
        external_param = self.internal_to_external(
            getattr(self.module, self.parameter_name)
        )
        # Create a new array with the updated values
        updated_external = external_param.at[key].set(value)
        # Transform back to internal representation
        internal_param = self.external_to_internal(updated_external)
        # Update the parameter in the module
        setattr(self.module, self.parameter_name, internal_param)

    def clone(self):
        # JAX arrays are immutable, so we don't need to clone
        return self.internal_to_external(getattr(self.module, self.parameter_name))

    @property
    def shape(self):
        return self.internal_to_external(
            getattr(self.module, self.parameter_name)
        ).shape

    @property
    def ndim(self):
        return self.internal_to_external(getattr(self.module, self.parameter_name)).ndim

    @property
    def grad(self):
        # In JAX, gradients are handled differently, typically through grad functions
        # This is a placeholder - gradient access would depend on how your optimization is set up
        raise NotImplementedError(
            "Gradient access in JAX differs from PyTorch. Use jax.grad or related functions instead."
        )

    def __getattr__(self, name: str):
        # Forward attribute access to the underlying array
        return getattr(getattr(self.module, self.parameter_name), name)


class Identity(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, x: jax.Array):
        return x
