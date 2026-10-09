"""Inception V3 architecture in Flax/JAX with Global Average Pooling."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from models.inception.blocks import (
    ConvBlock,
    InceptionA,
    InceptionB,
    InceptionC,
    InceptionD,
    InceptionE,
)


class InceptionV3(nn.Module):
    """Inception V3 backbone for Federated Learning.

    Features factorized asymmetric convolutions, grid-size reduction modules,
    and expanded filter banks, followed by Global Average Pooling.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default is 10.
    norm : str
        Normalization type ('group', 'batch', 'none'). Default is 'group'.
    """

    num_classes: int = 10
    norm: str = "group"

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = True, **kwargs: Any
    ) -> jnp.ndarray:
        # Stem
        x = ConvBlock(features=32, kernel_size=(3, 3), norm=self.norm)(x, train=train)
        x = ConvBlock(features=32, kernel_size=(3, 3), norm=self.norm)(x, train=train)
        x = ConvBlock(features=64, kernel_size=(3, 3), norm=self.norm)(x, train=train)
        x = ConvBlock(features=80, kernel_size=(1, 1), norm=self.norm)(x, train=train)
        x = ConvBlock(features=192, kernel_size=(3, 3), norm=self.norm)(x, train=train)

        # Inception-A stage (3 blocks)
        x = InceptionA(pool_features=32, norm=self.norm)(x, train=train)
        x = InceptionA(pool_features=64, norm=self.norm)(x, train=train)
        x = InceptionA(pool_features=64, norm=self.norm)(x, train=train)

        # Inception-B stage (reduction block)
        x = InceptionB(norm=self.norm)(x, train=train)

        # Inception-C stage (4 blocks)
        x = InceptionC(channels_7x7=128, norm=self.norm)(x, train=train)
        x = InceptionC(channels_7x7=160, norm=self.norm)(x, train=train)
        x = InceptionC(channels_7x7=160, norm=self.norm)(x, train=train)
        x = InceptionC(channels_7x7=192, norm=self.norm)(x, train=train)

        # Inception-D stage (reduction block)
        x = InceptionD(norm=self.norm)(x, train=train)

        # Inception-E stage (2 blocks)
        x = InceptionE(norm=self.norm)(x, train=train)
        x = InceptionE(norm=self.norm)(x, train=train)

        # Global Average Pooling and classification head
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=self.num_classes)(x)
        return x
