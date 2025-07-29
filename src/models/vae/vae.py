"""
Variational Autoencoder implementation using Flax NNX API.
"""

from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.nnx import initializers
from jax.sharding import Mesh


@dataclass
class VAEConfig:
    """Configuration for VAE model."""

    # Image configuration
    image_size: int = 28  # Height and width of square images
    channels: int = 3

    hidden_dims: Tuple[int, ...] = (32, 64, 128, 256)
    latent_dim: int = 128
    activation: str = "relu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = True

    kernel_size: tuple[int, int] = (4, 4)
    strides: tuple[int, int] = (2, 2)
    padding: tuple[int, int] = (1, 1)

    @property
    def input_dim(self) -> int:
        return self.image_size * self.image_size * self.channels


class ConvBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, int],
        strides: tuple[int, int],
        padding: tuple[int, int],
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        activation=partial(jax.nn.leaky_relu, negative_slope=0.2),
    ):
        self.conv = nnx.Conv(
            in_features,
            out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=(None, None, None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.norm = nnx.BatchNorm(
            out_features,
            rngs=rngs,
            dtype=param_dtype,
            param_dtype=param_dtype,
            scale_init=nnx.with_partitioning(
                initializers.ones_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.activation = activation

    def __call__(self, x: jax.Array):
        out = self.conv(x).astype(jnp.float32)
        return self.activation(self.norm(out))


class ConvTransposeBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, int],
        strides: tuple[int, int],
        padding: tuple[int, int],
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        activation=partial(jax.nn.leaky_relu, negative_slope=0.2),
    ):
        self.conv = nnx.ConvTranspose(
            in_features,
            out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=(None, None, None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.norm = nnx.BatchNorm(
            out_features,
            rngs=rngs,
            dtype=param_dtype,
            param_dtype=param_dtype,
            scale_init=nnx.with_partitioning(
                initializers.ones_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.activation = activation

    def __call__(self, x: jax.Array):
        out = self.conv(x).astype(jnp.float32)
        return self.activation(self.norm(out))


class Encoder(nnx.Module):
    """Encoder network that maps input to latent distribution parameters."""

    def __init__(
        self,
        config: VAEConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.config = config

        hidden_dims = config.hidden_dims
        shifted_dims = hidden_dims[:-1]
        shifted_dims = (config.channels,) + shifted_dims

        layers: list[ConvBlock] = []
        for in_dim, out_dim in zip(shifted_dims, hidden_dims):
            layers.append(
                ConvBlock(
                    in_dim,
                    out_dim,
                    kernel_size=config.kernel_size,
                    strides=config.strides,
                    padding=config.padding,
                    rngs=rngs,
                    dtype=dtype,
                    param_dtype=param_dtype,
                )
            )

        self.layers = layers

        post_conv_dim = 256 * 4 * 4

        # Output layers for mean and log variance
        Linear = partial(
            nnx.Linear,
            post_conv_dim,
            config.latent_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.mean_layer = Linear()
        self.logvar_layer = Linear()

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        for layer in self.layers:
            x = layer(x)

        x = x.reshape(x.shape[0], -1)

        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)

        return mean, logvar


class Decoder(nnx.Module):
    """Decoder network that maps latent codes back to input space."""

    def __init__(
        self,
        config: VAEConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.config = config

        post_conv_dim = 256 * 4 * 4
        self.inner_proj = nnx.Linear(
            config.latent_dim,
            post_conv_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        hidden_dims = list(reversed(config.hidden_dims))
        shifted_dims = hidden_dims[1:]
        shifted_dims.append(config.channels)

        layers: list[ConvTransposeBlock] = []

        for in_dim, out_dim in zip(hidden_dims, shifted_dims):
            layers.append(
                ConvTransposeBlock(
                    in_dim,
                    out_dim,
                    kernel_size=config.kernel_size,
                    strides=config.strides,
                    padding=config.padding,
                    rngs=rngs,
                    dtype=dtype,
                    param_dtype=param_dtype,
                )
            )

        self.layers = layers

    def __call__(self, z: jax.Array):
        reconstruction = self.inner_proj(z)
        reconstruction = reconstruction.reshape(-1, 256, 4, 4)

        for layer in self.layers:
            reconstruction = layer(reconstruction)

        return jax.nn.tanh(reconstruction)


class VAE(nnx.Module):
    """Variational Autoencoder using Flax NNX."""

    def __init__(
        self,
        config: VAEConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.config = config
        self.encoder = Encoder(
            config,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.decoder = Decoder(
            config,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def encode(self, x: jax.Array):
        """Encode input to latent distribution parameters."""
        return self.encoder(x)

    def reparameterize(
        self,
        mean: jax.Array,
        logvar: jax.Array,
        *,
        rngs: nnx.Rngs,
    ):
        """Reparameterization trick to sample from latent distribution."""

        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rngs.sample(), mean.shape)
        return mean + eps * std

    def decode(self, z: jax.Array):
        """Decode latent codes to reconstructed input."""
        return self.decoder(z)

    def __call__(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs,
        training=False,
    ):
        mean, logvar = self.encode(x)
        z = jax.lax.cond(
            training,
            lambda: self.reparameterize(mean, logvar, rngs=rngs),
            lambda: mean,
        )

        reconstruction = self.decode(z)

        return reconstruction, mean, logvar

    def generate(
        self,
        num_samples: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Generate new samples from the prior distribution.

        Args:
            num_samples: Number of samples to generate
            rngs: Random number generators
            return_as_images: If True, return images in proper shape, otherwise flat

        Returns:
            Generated samples as flat vectors or reshaped images
        """
        z = jax.random.normal(rngs.sample(), (num_samples, self.config.latent_dim))
        return self.decode(z)


def vae_loss(
    reconstruction: jax.Array,
    x: jax.Array,
    mean: jax.Array,
    logvar: jax.Array,
    beta: float = 1.0,
):
    """VAE loss function with KL divergence and reconstruction loss.

    Args:
        reconstruction: Reconstructed input (in range [-1, 1] from tanh)
        x: Original input (in range [-1, 1])
        mean: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence term (beta-VAE)

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss)
    """
    # Reconstruction loss - Mean Squared Error works well with [-1, 1] range
    reconstruction_loss = optax.squared_error(reconstruction, x).mean()

    # KL divergence loss

    kl_loss = -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1)
    kl_loss = jnp.mean(kl_loss)

    # Total loss
    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss
