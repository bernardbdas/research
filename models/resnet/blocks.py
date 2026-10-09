"""ResNet residual building blocks."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp


def get_norm_layer(norm: str, features: int, train: bool = True) -> Any:
    """Return normalization layer based on norm type.

    Defaults to GroupNorm for federated learning stability (no running statistics,
    pure parameter trees, robust under non-IID client heterogeneity).
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


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock with two 3x3 convolutions and skip connection."""

    features: int
    stride: int = 1
    norm: str = "group"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x

        # Conv 1
        y = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
        )(x)
        y = get_norm_layer(self.norm, self.features, train=train)(y)
        y = nn.relu(y)

        # Conv 2
        y = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(y)
        y = get_norm_layer(self.norm, self.features, train=train)(y)

        # Projection shortcut if input dimensions change
        if residual.shape[-1] != self.features or self.stride != 1:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
            )(residual)
            residual = get_norm_layer(self.norm, self.features, train=train)(residual)

        return nn.relu(residual + y)


class Bottleneck(nn.Module):
    """Bottleneck residual block for deeper ResNet architectures (ResNet-50, 101, 152).

    Uses a 3-layer stack: 1x1 conv (compression) -> 3x3 conv -> 1x1 conv (expansion by 4x).
    """

    features: int
    stride: int = 1
    norm: str = "group"
    expansion: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        out_features = self.features * self.expansion

        # 1. 1x1 Conv (compression / reduction)
        y = nn.Conv(
            features=self.features,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        y = get_norm_layer(self.norm, self.features, train=train)(y)
        y = nn.relu(y)

        # 2. 3x3 Conv
        y = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
        )(y)
        y = get_norm_layer(self.norm, self.features, train=train)(y)
        y = nn.relu(y)

        # 3. 1x1 Conv (expansion)
        y = nn.Conv(
            features=out_features,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(y)
        y = get_norm_layer(self.norm, out_features, train=train)(y)

        # Projection shortcut if input dimensions change
        if residual.shape[-1] != out_features or self.stride != 1:
            residual = nn.Conv(
                features=out_features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
            )(residual)
            residual = get_norm_layer(self.norm, out_features, train=train)(residual)

        return nn.relu(residual + y)
