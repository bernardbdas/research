"""Inception V3 building blocks in Flax/JAX."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp


def get_norm_layer(norm: str, features: int, train: bool = True) -> Any:
    """Return normalization layer based on norm type.

    Defaults to GroupNorm for federated learning stability (pure parameter trees).
    """
    if norm == "group":
        num_groups = min(16, features)
        while features % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        return nn.GroupNorm(num_groups=num_groups)
    if norm == "batch":
        return nn.BatchNorm(use_running_average=not train)
    if norm == "none":
        return lambda x: x
    raise ValueError(f"Unsupported norm: '{norm}'. Choose 'group', 'batch', or 'none'.")


class ConvBlock(nn.Module):
    """Convolution + Normalization + ReLU block."""

    features: int
    kernel_size: tuple[int, int]
    strides: tuple[int, int] = (1, 1)
    padding: str = "SAME"
    norm: str = "group"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=False,
        )(x)
        x = get_norm_layer(self.norm, self.features, train=train)(x)
        return nn.relu(x)


class InceptionA(nn.Module):
    """Inception-A block: factorizes 5x5 convolutions into two 3x3 convolutions."""

    pool_features: int
    norm: str = "group"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Branch 1
        b1 = ConvBlock(features=64, kernel_size=(1, 1), norm=self.norm)(x, train=train)

        # Branch 2
        b2 = ConvBlock(features=48, kernel_size=(1, 1), norm=self.norm)(x, train=train)
        b2 = ConvBlock(features=64, kernel_size=(3, 3), norm=self.norm)(b2, train=train)

        # Branch 3
        b3 = ConvBlock(features=64, kernel_size=(1, 1), norm=self.norm)(x, train=train)
        b3 = ConvBlock(features=96, kernel_size=(3, 3), norm=self.norm)(b3, train=train)
        b3 = ConvBlock(features=96, kernel_size=(3, 3), norm=self.norm)(b3, train=train)

        # Branch 4
        b4 = nn.avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")
        b4 = ConvBlock(features=self.pool_features, kernel_size=(1, 1), norm=self.norm)(
            b4, train=train
        )

        return jnp.concatenate([b1, b2, b3, b4], axis=-1)


class InceptionB(nn.Module):
    """Inception-B block: grid reduction block (stride 2)."""

    norm: str = "group"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Branch 1
        b1 = ConvBlock(
            features=384,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            norm=self.norm,
        )(x, train=train)

        # Branch 2
        b2 = ConvBlock(features=64, kernel_size=(1, 1), norm=self.norm)(x, train=train)
        b2 = ConvBlock(features=96, kernel_size=(3, 3), norm=self.norm)(b2, train=train)
        b2 = ConvBlock(
            features=96,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            norm=self.norm,
        )(b2, train=train)

        # Branch 3
        b3 = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        return jnp.concatenate([b1, b2, b3], axis=-1)


class InceptionC(nn.Module):
    """Inception-C block: factorizes n x n into 1 x n and n x 1 convolutions."""

    channels_7x7: int = 128
    norm: str = "group"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        c7 = self.channels_7x7

        # Branch 1
        b1 = ConvBlock(features=192, kernel_size=(1, 1), norm=self.norm)(x, train=train)

        # Branch 2
        b2 = ConvBlock(features=c7, kernel_size=(1, 1), norm=self.norm)(x, train=train)
        b2 = ConvBlock(features=c7, kernel_size=(1, 7), norm=self.norm)(b2, train=train)
        b2 = ConvBlock(features=192, kernel_size=(7, 1), norm=self.norm)(
            b2, train=train
        )

        # Branch 3
        b3 = ConvBlock(features=c7, kernel_size=(1, 1), norm=self.norm)(x, train=train)
        b3 = ConvBlock(features=c7, kernel_size=(7, 1), norm=self.norm)(b3, train=train)
        b3 = ConvBlock(features=c7, kernel_size=(1, 7), norm=self.norm)(b3, train=train)
        b3 = ConvBlock(features=c7, kernel_size=(7, 1), norm=self.norm)(b3, train=train)
        b3 = ConvBlock(features=192, kernel_size=(1, 7), norm=self.norm)(
            b3, train=train
        )

        # Branch 4
        b4 = nn.avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")
        b4 = ConvBlock(features=192, kernel_size=(1, 1), norm=self.norm)(
            b4, train=train
        )

        return jnp.concatenate([b1, b2, b3, b4], axis=-1)


class InceptionD(nn.Module):
    """Inception-D block: second grid reduction block (stride 2)."""

    norm: str = "group"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Branch 1
        b1 = ConvBlock(features=192, kernel_size=(1, 1), norm=self.norm)(x, train=train)
        b1 = ConvBlock(
            features=320,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            norm=self.norm,
        )(b1, train=train)

        # Branch 2
        b2 = ConvBlock(features=192, kernel_size=(1, 1), norm=self.norm)(x, train=train)
        b2 = ConvBlock(features=192, kernel_size=(1, 7), norm=self.norm)(
            b2, train=train
        )
        b2 = ConvBlock(features=192, kernel_size=(7, 1), norm=self.norm)(
            b2, train=train
        )
        b2 = ConvBlock(
            features=192,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            norm=self.norm,
        )(b2, train=train)

        # Branch 3
        b3 = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        return jnp.concatenate([b1, b2, b3], axis=-1)


class InceptionE(nn.Module):
    """Inception-E block: expanded filter bank with parallel 1x3 and 3x1 convolutions."""

    norm: str = "group"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Branch 1
        b1 = ConvBlock(features=320, kernel_size=(1, 1), norm=self.norm)(x, train=train)

        # Branch 2
        b2 = ConvBlock(features=384, kernel_size=(1, 1), norm=self.norm)(x, train=train)
        b2_a = ConvBlock(features=384, kernel_size=(1, 3), norm=self.norm)(
            b2, train=train
        )
        b2_b = ConvBlock(features=384, kernel_size=(3, 1), norm=self.norm)(
            b2, train=train
        )
        b2 = jnp.concatenate([b2_a, b2_b], axis=-1)

        # Branch 3
        b3 = ConvBlock(features=448, kernel_size=(1, 1), norm=self.norm)(x, train=train)
        b3 = ConvBlock(features=384, kernel_size=(3, 3), norm=self.norm)(
            b3, train=train
        )
        b3_a = ConvBlock(features=384, kernel_size=(1, 3), norm=self.norm)(
            b3, train=train
        )
        b3_b = ConvBlock(features=384, kernel_size=(3, 1), norm=self.norm)(
            b3, train=train
        )
        b3 = jnp.concatenate([b3_a, b3_b], axis=-1)

        # Branch 4
        b4 = nn.avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")
        b4 = ConvBlock(features=192, kernel_size=(1, 1), norm=self.norm)(
            b4, train=train
        )

        return jnp.concatenate([b1, b2, b3, b4], axis=-1)
