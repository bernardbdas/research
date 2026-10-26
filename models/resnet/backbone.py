"""ResNet architectures in Flax/JAX with Global Average Pooling."""

from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from models.resnet.blocks import BasicBlock, Bottleneck, get_norm_layer


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
    norm : str
        Normalization type ('group', 'batch', 'none'). Default is 'group'.
    """

    stage_sizes: Sequence[int] = (2, 2, 2, 2)
    num_classes: int = 10
    norm: str = "group"

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = True, **kwargs: Any
    ) -> jnp.ndarray:
        # Initial convolution (preserves spatial resolution for small & large inputs)
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = get_norm_layer(self.norm, 64, train=train)(x)
        x = nn.relu(x)

        # Stages
        channels = [64, 128, 256, 512]
        for stage_idx, (num_blocks, feat) in enumerate(
            zip(self.stage_sizes, channels, strict=True)
        ):
            stride = 1 if stage_idx == 0 else 2
            x = BasicBlock(features=feat, stride=stride, norm=self.norm)(x, train=train)
            for _ in range(1, num_blocks):
                x = BasicBlock(features=feat, stride=1, norm=self.norm)(x, train=train)

        # Global Average Pooling and classification head
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=self.num_classes)(x)
        return x


class ResNet18(ResNet):
    """ResNet-18 backbone with Global Average Pooling."""

    stage_sizes: Sequence[int] = (2, 2, 2, 2)
    num_classes: int = 10
    norm: str = "group"


class ResNet9(nn.Module):
    """High-speed ResNet-9 backbone optimized for CIFAR-10 and federated edge clients.

    Incorporates residual skip connections across 3 stages with GroupNorm, reaching
    >90% CIFAR-10 test accuracy rapidly with ~6.5M parameters and pure parameter trees.
    """

    num_classes: int = 10
    norm: str = "group"

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = True, **kwargs: Any
    ) -> jnp.ndarray:
        # Prep: Conv 64
        x = nn.Conv(64, (3, 3), padding="SAME", use_bias=False)(x)
        x = get_norm_layer(self.norm, 64, train=train)(x)
        x = nn.relu(x)

        # Layer 1: Conv 128 -> MaxPool -> Residual Block (128)
        x = nn.Conv(128, (3, 3), padding="SAME", use_bias=False)(x)
        x = get_norm_layer(self.norm, 128, train=train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = BasicBlock(features=128, stride=1, norm=self.norm)(x, train=train)

        # Layer 2: Conv 256 -> MaxPool
        x = nn.Conv(256, (3, 3), padding="SAME", use_bias=False)(x)
        x = get_norm_layer(self.norm, 256, train=train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Layer 3: Conv 512 -> MaxPool -> Residual Block (512)
        x = nn.Conv(512, (3, 3), padding="SAME", use_bias=False)(x)
        x = get_norm_layer(self.norm, 512, train=train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = BasicBlock(features=512, stride=1, norm=self.norm)(x, train=train)

        # Global Average Pooling and classification head
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=self.num_classes)(x)
        return x


class ResNet50(nn.Module):
    """ResNet-50 backbone with Bottleneck residual blocks and Global Average Pooling.

    Features 4 stages with [3, 4, 6, 3] Bottleneck blocks and GroupNorm (~23.5M parameters),
    producing 2048-dimensional representations before the classification head.

    Parameters
    ----------
    stage_sizes : Sequence[int]
        Number of blocks in each stage ((3, 4, 6, 3) for ResNet-50). Default is (3, 4, 6, 3).
    num_classes : int
        Number of output classes. Default is 10.
    norm : str
        Normalization type ('group', 'batch', 'none'). Default is 'group'.
    """

    stage_sizes: Sequence[int] = (3, 4, 6, 3)
    num_classes: int = 10
    norm: str = "group"

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = True, **kwargs: Any
    ) -> jnp.ndarray:
        # Initial convolution
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = get_norm_layer(self.norm, 64, train=train)(x)
        x = nn.relu(x)

        # 4 stages with Bottleneck blocks (expansion=4)
        channels = [64, 128, 256, 512]
        for stage_idx, (num_blocks, feat) in enumerate(
            zip(self.stage_sizes, channels, strict=True)
        ):
            stride = 1 if stage_idx == 0 else 2
            x = Bottleneck(features=feat, stride=stride, norm=self.norm)(x, train=train)
            for _ in range(1, num_blocks):
                x = Bottleneck(features=feat, stride=1, norm=self.norm)(x, train=train)

        # Global Average Pooling and classification head
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=self.num_classes)(x)
        return x
