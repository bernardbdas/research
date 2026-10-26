"""DenseNet building blocks in Flax/JAX."""

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


class DenseLayer(nn.Module):
    """Bottleneck DenseLayer: BN/GN -> ReLU -> 1x1 Conv (4k) -> BN/GN -> ReLU -> 3x3 Conv (k)."""

    growth_rate: int = 32
    bn_size: int = 4
    norm: str = "group"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        in_channels = x.shape[-1]
        inter_channels = self.bn_size * self.growth_rate

        # 1. 1x1 Conv (bottleneck compression)
        y = get_norm_layer(self.norm, in_channels, train=train)(x)
        y = nn.relu(y)
        y = nn.Conv(
            features=inter_channels, kernel_size=(1, 1), padding="SAME", use_bias=False
        )(y)

        # 2. 3x3 Conv (feature growth)
        y = get_norm_layer(self.norm, inter_channels, train=train)(y)
        y = nn.relu(y)
        y = nn.Conv(
            features=self.growth_rate,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
        )(y)

        # Densely concatenate new features with previous feature maps
        return jnp.concatenate([x, y], axis=-1)


class DenseBlock(nn.Module):
    """DenseBlock chaining multiple DenseLayers with dense feature concatenation."""

    num_layers: int
    growth_rate: int = 32
    bn_size: int = 4
    norm: str = "group"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        for _ in range(self.num_layers):
            x = DenseLayer(
                growth_rate=self.growth_rate,
                bn_size=self.bn_size,
                norm=self.norm,
            )(x, train=train)
        return x


class TransitionLayer(nn.Module):
    """Transition layer between DenseBlocks for feature compression and spatial downsampling."""

    out_features: int
    norm: str = "group"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        in_channels = x.shape[-1]
        x = get_norm_layer(self.norm, in_channels, train=train)(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=self.out_features,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        return x
