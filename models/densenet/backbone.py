"""DenseNet architectures in Flax/JAX with Global Average Pooling."""

from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from models.densenet.blocks import DenseBlock, TransitionLayer, get_norm_layer


class DenseNet(nn.Module):
    """Generic, data-agnostic DenseNet backbone for Federated Learning.

    Parameters
    ----------
    growth_rate : int
        Number of filters added in each layer (k). Default is 32.
    block_config : Sequence[int]
        Number of layers in each of the 4 dense blocks. Default is (6, 12, 32, 32) for DenseNet-169.
    init_features : int
        Number of filters in the initial convolution. Default is 64.
    bn_size : int
        Bottleneck compression factor for 1x1 convs. Default is 4.
    drop_rate : float
        Dropout probability in DenseLayers. Default is 0.0.
    num_classes : int
        Number of classification output classes. Default is 10.
    norm : str
        Normalization type ('group', 'batch', 'none'). Default is 'group'.
    """

    growth_rate: int = 32
    block_config: Sequence[int] = (6, 12, 32, 32)
    init_features: int = 64
    bn_size: int = 4
    drop_rate: float = 0.0
    num_classes: int = 10
    norm: str = "group"

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = True, **kwargs: Any
    ) -> jnp.ndarray:
        # Initial convolution
        x = nn.Conv(
            features=self.init_features,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
        )(x)
        x = get_norm_layer(self.norm, self.init_features, train=train)(x)
        x = nn.relu(x)

        num_features = self.init_features

        # DenseBlocks and TransitionLayers
        for i, num_layers in enumerate(self.block_config):
            x = DenseBlock(
                num_layers=num_layers,
                growth_rate=self.growth_rate,
                bn_size=self.bn_size,
                norm=self.norm,
            )(x, train=train)
            num_features = num_features + num_layers * self.growth_rate

            # Transition layer (halves feature channels and downsamples spatially)
            if i != len(self.block_config) - 1:
                trans_features = num_features // 2
                x = TransitionLayer(out_features=trans_features, norm=self.norm)(
                    x, train=train
                )
                num_features = trans_features

        # Final Normalization and Activation
        x = get_norm_layer(self.norm, num_features, train=train)(x)
        x = nn.relu(x)

        # Global Average Pooling and classification head
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=self.num_classes)(x)
        return x


class DenseNet169(DenseNet):
    """DenseNet-169 backbone with (6, 12, 32, 32) block configuration."""

    growth_rate: int = 32
    block_config: Sequence[int] = (6, 12, 32, 32)
    init_features: int = 64
    bn_size: int = 4
    num_classes: int = 10
    norm: str = "group"
