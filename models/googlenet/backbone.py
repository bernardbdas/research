"""GoogLeNet (Inception v1) architecture in Flax/JAX with Global Average Pooling."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from models.googlenet.blocks import InceptionBlock, get_norm_layer


class GoogLeNet(nn.Module):
    """GoogLeNet (Inception v1) backbone for Federated Learning.

    Uses 9 Inception modules organized across 3 stages, followed by
    Global Average Pooling and a classification head.

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
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME", use_bias=False)(x)
        x = get_norm_layer(self.norm, 64, train=train)(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(1, 1), padding="SAME", use_bias=False)(x)
        x = get_norm_layer(self.norm, 64, train=train)(x)
        x = nn.relu(x)

        x = nn.Conv(features=192, kernel_size=(3, 3), padding="SAME", use_bias=False)(x)
        x = get_norm_layer(self.norm, 192, train=train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # Inception Stage 3
        # Inception 3a: 64, 96, 128, 16, 32, 32 -> out: 256
        x = InceptionBlock(64, 96, 128, 16, 32, 32, norm=self.norm)(x, train=train)
        # Inception 3b: 128, 128, 192, 32, 96, 64 -> out: 480
        x = InceptionBlock(128, 128, 192, 32, 96, 64, norm=self.norm)(x, train=train)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # Inception Stage 4
        # Inception 4a: 192, 96, 208, 16, 48, 64 -> out: 512
        x = InceptionBlock(192, 96, 208, 16, 48, 64, norm=self.norm)(x, train=train)
        # Inception 4b: 160, 112, 224, 24, 64, 64 -> out: 512
        x = InceptionBlock(160, 112, 224, 24, 64, 64, norm=self.norm)(x, train=train)
        # Inception 4c: 128, 128, 256, 24, 64, 64 -> out: 512
        x = InceptionBlock(128, 128, 256, 24, 64, 64, norm=self.norm)(x, train=train)
        # Inception 4d: 112, 144, 288, 32, 64, 64 -> out: 528
        x = InceptionBlock(112, 144, 288, 32, 64, 64, norm=self.norm)(x, train=train)
        # Inception 4e: 256, 160, 320, 32, 128, 128 -> out: 832
        x = InceptionBlock(256, 160, 320, 32, 128, 128, norm=self.norm)(x, train=train)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # Inception Stage 5
        # Inception 5a: 256, 160, 320, 32, 128, 128 -> out: 832
        x = InceptionBlock(256, 160, 320, 32, 128, 128, norm=self.norm)(x, train=train)
        # Inception 5b: 384, 192, 384, 48, 128, 128 -> out: 1024
        x = InceptionBlock(384, 192, 384, 48, 128, 128, norm=self.norm)(x, train=train)

        # Global Average Pooling and classification head
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=self.num_classes)(x)
        return x
