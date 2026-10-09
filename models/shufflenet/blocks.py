"""ShuffleNet V2 building blocks in Flax/JAX."""

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


def channel_shuffle(x: jnp.ndarray, groups: int = 2) -> jnp.ndarray:
    """Channel shuffle operator to enable cross-group feature communication.

    Reshapes (B, H, W, G * C') to (B, H, W, G, C'), transposes the last two dims,
    and flattens back to (B, H, W, C).
    """
    b, h, w, c = x.shape
    channels_per_group = c // groups
    x = x.reshape((b, h, w, groups, channels_per_group))
    x = jnp.transpose(x, (0, 1, 2, 4, 3))
    return x.reshape((b, h, w, c))


class ShuffleV2Block(nn.Module):
    """ShuffleNet V2 inverted residual block with channel split and channel shuffle.

    Parameters
    ----------
    out_channels : int
        Total output channel dimension after concatenation.
    stride : int
        Convolution stride (1 or 2). Default is 1.
    norm : str
        Normalization type ('group', 'batch', 'none'). Default is 'group'.
    """

    out_channels: int
    stride: int = 1
    norm: str = "group"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        branch_channels = self.out_channels // 2

        if self.stride == 1:
            # Channel Split: split input equally into two halves
            c1, c2 = jnp.split(x, 2, axis=-1)

            # Branch 2: 1x1 conv -> 3x3 depthwise conv -> 1x1 conv
            y2 = nn.Conv(
                features=branch_channels,
                kernel_size=(1, 1),
                padding="SAME",
                use_bias=False,
            )(c2)
            y2 = get_norm_layer(self.norm, branch_channels, train=train)(y2)
            y2 = nn.relu(y2)

            y2 = nn.Conv(
                features=branch_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                feature_group_count=branch_channels,
                use_bias=False,
            )(y2)
            y2 = get_norm_layer(self.norm, branch_channels, train=train)(y2)

            y2 = nn.Conv(
                features=branch_channels,
                kernel_size=(1, 1),
                padding="SAME",
                use_bias=False,
            )(y2)
            y2 = get_norm_layer(self.norm, branch_channels, train=train)(y2)
            y2 = nn.relu(y2)

            out = jnp.concatenate([c1, y2], axis=-1)
        else:
            # Stride == 2: Downsampling, both branches process the full input
            # Branch 1: 3x3 depthwise conv -> 1x1 conv
            y1 = nn.Conv(
                features=x.shape[-1],
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                feature_group_count=x.shape[-1],
                use_bias=False,
            )(x)
            y1 = get_norm_layer(self.norm, x.shape[-1], train=train)(y1)

            y1 = nn.Conv(
                features=branch_channels,
                kernel_size=(1, 1),
                padding="SAME",
                use_bias=False,
            )(y1)
            y1 = get_norm_layer(self.norm, branch_channels, train=train)(y1)
            y1 = nn.relu(y1)

            # Branch 2: 1x1 conv -> 3x3 depthwise conv -> 1x1 conv
            y2 = nn.Conv(
                features=branch_channels,
                kernel_size=(1, 1),
                padding="SAME",
                use_bias=False,
            )(x)
            y2 = get_norm_layer(self.norm, branch_channels, train=train)(y2)
            y2 = nn.relu(y2)

            y2 = nn.Conv(
                features=branch_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                feature_group_count=branch_channels,
                use_bias=False,
            )(y2)
            y2 = get_norm_layer(self.norm, branch_channels, train=train)(y2)

            y2 = nn.Conv(
                features=branch_channels,
                kernel_size=(1, 1),
                padding="SAME",
                use_bias=False,
            )(y2)
            y2 = get_norm_layer(self.norm, branch_channels, train=train)(y2)
            y2 = nn.relu(y2)

            out = jnp.concatenate([y1, y2], axis=-1)

        return channel_shuffle(out, groups=2)
