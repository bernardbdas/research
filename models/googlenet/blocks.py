"""GoogLeNet (Inception v1) building blocks in Flax/JAX."""

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


class InceptionBlock(nn.Module):
    """Canonical Inception building block with 4 parallel convolutional branches.

    Parameters
    ----------
    ch1x1 : int
        Number of output filters for branch 1 (1x1 conv).
    ch3x3_red : int
        Number of reduction filters for branch 2 (1x1 conv before 3x3).
    ch3x3 : int
        Number of output filters for branch 2 (3x3 conv).
    ch5x5_red : int
        Number of reduction filters for branch 3 (1x1 conv before 5x5).
    ch5x5 : int
        Number of output filters for branch 3 (5x5 / two 3x3 convs).
    pool_proj : int
        Number of output filters for branch 4 (1x1 conv after 3x3 max pooling).
    norm : str
        Normalization type ('group', 'batch', 'none'). Default is 'group'.
    """

    ch1x1: int
    ch3x3_red: int
    ch3x3: int
    ch5x5_red: int
    ch5x5: int
    pool_proj: int
    norm: str = "group"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Branch 1: 1x1 conv
        b1 = nn.Conv(
            features=self.ch1x1, kernel_size=(1, 1), padding="SAME", use_bias=False
        )(x)
        b1 = get_norm_layer(self.norm, self.ch1x1, train=train)(b1)
        b1 = nn.relu(b1)

        # Branch 2: 1x1 conv -> 3x3 conv
        b2 = nn.Conv(
            features=self.ch3x3_red, kernel_size=(1, 1), padding="SAME", use_bias=False
        )(x)
        b2 = get_norm_layer(self.norm, self.ch3x3_red, train=train)(b2)
        b2 = nn.relu(b2)
        b2 = nn.Conv(
            features=self.ch3x3, kernel_size=(3, 3), padding="SAME", use_bias=False
        )(b2)
        b2 = get_norm_layer(self.norm, self.ch3x3, train=train)(b2)
        b2 = nn.relu(b2)

        # Branch 3: 1x1 conv -> 5x5 (factorized as two 3x3) conv
        b3 = nn.Conv(
            features=self.ch5x5_red, kernel_size=(1, 1), padding="SAME", use_bias=False
        )(x)
        b3 = get_norm_layer(self.norm, self.ch5x5_red, train=train)(b3)
        b3 = nn.relu(b3)
        b3 = nn.Conv(
            features=self.ch5x5, kernel_size=(3, 3), padding="SAME", use_bias=False
        )(b3)
        b3 = get_norm_layer(self.norm, self.ch5x5, train=train)(b3)
        b3 = nn.relu(b3)

        # Branch 4: 3x3 max pool -> 1x1 conv
        b4 = nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")
        b4 = nn.Conv(
            features=self.pool_proj, kernel_size=(1, 1), padding="SAME", use_bias=False
        )(b4)
        b4 = get_norm_layer(self.norm, self.pool_proj, train=train)(b4)
        b4 = nn.relu(b4)

        return jnp.concatenate([b1, b2, b3, b4], axis=-1)
