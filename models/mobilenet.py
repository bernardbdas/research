"""MobileNetV2 architecture in Flax/JAX with Global Average Pooling."""

import flax.linen as nn
import jax.numpy as jnp


class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block with linear bottleneck."""

    in_features: int
    out_features: int
    stride: int
    expand_ratio: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        hidden_dim = self.in_features * self.expand_ratio
        use_residual = self.stride == 1 and self.in_features == self.out_features
        residual = x

        # 1. 1x1 Conv (Expansion)
        if self.expand_ratio != 1:
            x = nn.Conv(features=hidden_dim, kernel_size=(1, 1), use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.relu6(x)

        # 2. 3x3 Depthwise Conv
        x = nn.Conv(
            features=hidden_dim,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            feature_group_count=hidden_dim,
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu6(x)

        # 3. 1x1 Conv (Linear Bottleneck Projection - no non-linearity)
        x = nn.Conv(features=self.out_features, kernel_size=(1, 1), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        if use_residual:
            return residual + x
        return x


class MobileNetV2(nn.Module):
    """Generic, data-agnostic MobileNetV2 backbone for Federated Learning.

    Uses Global Average Pooling across spatial dimensions, allowing it
    to operate on inputs of arbitrary spatial resolutions.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default is 10.
    width_mult : float
        Width multiplier for channel scaling (e.g. 1.0, 0.5). Default is 1.0.
    """

    num_classes: int = 10
    width_mult: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Configuration: [t (expand_ratio), c (channels), n (repeats), s (stride)]
        inverted_residual_setting = [
            (1, 16, 1, 1),
            (6, 24, 2, 1),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        # Initial Conv
        init_features = int(32 * self.width_mult)
        x = nn.Conv(
            features=init_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu6(x)

        in_channels = init_features
        for t, c, n, s in inverted_residual_setting:
            out_channels = int(c * self.width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                x = InvertedResidual(
                    in_features=in_channels,
                    out_features=out_channels,
                    stride=stride,
                    expand_ratio=t,
                )(x, train=train)
                in_channels = out_channels

        # Final 1x1 Conv
        last_features = int(1280 * self.width_mult) if self.width_mult > 1.0 else 1280
        x = nn.Conv(features=last_features, kernel_size=(1, 1), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu6(x)

        # Global Average Pooling and classifier
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=self.num_classes)(x)
        return x
