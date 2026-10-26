"""MobileNetV2 building blocks."""

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
