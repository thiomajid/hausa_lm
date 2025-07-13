# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano


import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn.normalization import _canonicalize_axes, _compute_stats, _normalize
from flax.typing import Axes, Dtype, Initializer
from jax.sharding import Mesh


def LayerNorm(
    num_features: int,
    mesh: Mesh,
    rngs: nnx.Rngs,
    use_scale: bool = True,
    use_bias: bool = False,
    epsilon: float = 1e-5,
    dtype=jnp.float32,
):
    return nnx.LayerNorm(
        num_features=num_features,
        use_scale=use_scale,
        use_bias=use_bias,
        rngs=rngs,
        dtype=dtype,
        param_dtype=dtype,
        epsilon=epsilon,
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


class MultiHeadLayerNorm(nnx.Module):
    """
    Applies a custom group normalization.
    The code is adapted from flax.nnx GroupNorm class to match the torch implementation.

    Args:
        x: A 4D tensor of shape (B, NH, S, DH) where:
            B: Batch size
            NH: Number of heads / num_groups
            S: Sequence length
            DH: Dimension per head
    """

    def __init__(
        self,
        num_features: int,
        residual_scale: bool = True,
        *,
        epsilon: float = 1e-5,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        use_bias: bool = False,
        use_scale: bool = True,
        bias_init: Initializer = nnx.initializers.zeros_init(),
        scale_init: Initializer = nnx.initializers.ones_init(),
        reduction_axes: tp.Optional[Axes] = None,
        axis_name: tp.Optional[str] = None,
        axis_index_groups: tp.Any = None,
        use_fast_variance: bool = True,
        rngs: nnx.Rngs,
    ):
        # initialization code adapted from flax.nnx.GroupNorm.__init__ implementation
        # num_groups and group_size are computed within __call__ method
        self.feature_axis = -1

        feature_shape = (num_features,)
        self.scale: nnx.Param[jax.Array] | None
        if use_scale:
            key = rngs.params()
            self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
        else:
            self.scale = None

        self.bias: nnx.Param[jax.Array] | None
        if use_bias:
            key = rngs.params()
            self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
        else:
            self.bias = None

        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.residual_scale = residual_scale
        self.bias_init = bias_init
        self.scale_init = scale_init
        self.reduction_axes = reduction_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = use_fast_variance

    @property
    def scale_proxy(self) -> tp.Optional[jnp.ndarray]:
        if self.scale is None:
            return None
        if self.residual_scale:
            return 1.0 + self.scale
        else:
            return self.scale

    def __call__(
        self,
        x: jnp.ndarray,
        mask: tp.Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        B, NH, S, DH = x.shape

        # Rearrange from (B, NH, S, DH) to (B, S, NH, DH)
        gn_in_1 = jnp.transpose(x, (0, 2, 1, 3))
        # Reshape to (B * S, NH * DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)

        num_groups = NH
        group_size = NH * DH // num_groups

        # Reshape for per-group statistics
        x = gn_in_2

        # Code adapted from flax.nnx.GroupNorm.__call__ implementation
        if self.reduction_axes is not None:
            reduction_axes = self.reduction_axes
        else:
            reduction_axes = list(range(1, x.ndim - 1)) + [-1]

        reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)

        group_shape = x.shape[:-1] + (num_groups, group_size)
        if mask is not None:
            mask = mask.reshape(mask.shape[:-1] + (num_groups, group_size))

        mean, var = _compute_stats(
            x.reshape(group_shape),
            list(reduction_axes[:-1]) + [-1],
            self.dtype,
            self.axis_name,
            self.axis_index_groups,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
        )
        mean = jnp.repeat(mean, group_size, axis=1)
        var = jnp.repeat(var, group_size, axis=1)
        normed = _normalize(
            x,
            mean,
            var,
            self.scale.value if self.scale else None,
            self.bias.value if self.bias else None,
            reduction_axes[:-1],
            (self.feature_axis,),
            self.dtype,
            self.epsilon,
        )

        # Reshape back to (B, S, NH, DH)
        normed = normed.reshape(B, S, NH, DH).transpose(0, 2, 1, 3)
        return normed
