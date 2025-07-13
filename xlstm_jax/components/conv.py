# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh


@dataclass(unsafe_hash=True, order=True)
class CausalConv1dConfig:
    feature_dim: int = None  # F
    kernel_size: int = 4
    causal_conv_bias: bool = True
    channel_mixing: bool = False
    conv1d_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.kernel_size >= 0, "kernel_size must be >= 0"


class CausalConv1d(nnx.Module):
    config_class = CausalConv1dConfig
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(
        self,
        config: CausalConv1dConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.groups = config.feature_dim
        self.kernel_size = config.kernel_size

        if config.channel_mixing:
            self.groups = 1

        if config.kernel_size == 0:
            self.conv = None
        else:
            # padding of this size assures temporal causality.
            self.pad = config.kernel_size - 1
            self.conv = nnx.Conv(
                in_features=config.feature_dim,
                out_features=config.feature_dim,
                kernel_size=(config.kernel_size,),
                padding=[(self.pad, 0)],
                feature_group_count=self.groups,
                use_bias=config.causal_conv_bias,
                rngs=rngs,
                dtype=dtype,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(
                    initializer=jax.nn.initializers.lecun_normal(),
                    sharding=(None, None, "tp"),
                    mesh=mesh,
                ),
                bias_init=nnx.with_partitioning(
                    initializer=nnx.initializers.zeros_init(),
                    sharding=("tp",),
                    mesh=mesh,
                ),
            )

    def __call__(self, x: jax.Array):
        return jax.lax.cond(
            self.kernel_size == 0,
            lambda: x,
            lambda: self.conv(x),
        )
