"""ResNet architectures in Flax/JAX with Global Average Pooling."""

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock with two 3x3 convolutions and skip connection."""

    features: int
    stride: int = 1

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
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)

        # Conv 2
        y = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(y)
        y = nn.BatchNorm(use_running_average=not train)(y)

        # Projection shortcut if input dimensions change
        if residual.shape[-1] != self.features or self.stride != 1:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        return nn.relu(residual + y)


class ResNet(nn.Module):
    """Generic, data-agnostic ResNet backbone for Federated Learning.

    Uses Global Average Pooling across spatial dimensions, allowing it to
    operate on inputs of arbitrary spatial resolutions.

    Parameters
    ----------
    stage_sizes : Sequence[int]
        Number of blocks in each stage (e.g. (2, 2, 2, 2) for ResNet-18).
    num_classes : int
        Number of output classes. Default is 10.
    """

    stage_sizes: Sequence[int] = (2, 2, 2, 2)
    num_classes: int = 10

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Initial convolution (preserves spatial resolution for small & large inputs)
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Stages
        channels = [64, 128, 256, 512]
        for stage_idx, (num_blocks, feat) in enumerate(
            zip(self.stage_sizes, channels, strict=True)
        ):
            stride = 1 if stage_idx == 0 else 2
            x = BasicBlock(features=feat, stride=stride)(x, train=train)
            for _ in range(1, num_blocks):
                x = BasicBlock(features=feat, stride=1)(x, train=train)

        # Global Average Pooling and classification head
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=self.num_classes)(x)
        return x


class ResNet18(ResNet):
    """ResNet-18 backbone with Global Average Pooling."""

    stage_sizes: Sequence[int] = (2, 2, 2, 2)
    num_classes: int = 10
