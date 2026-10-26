"""ShuffleNet V2 architecture in Flax/JAX with Global Average Pooling."""

from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from models.shufflenet.blocks import ShuffleV2Block, get_norm_layer


class ShuffleNetV2(nn.Module):
    """Generic ShuffleNet V2 backbone for Federated Learning and Edge Vision.

    Parameters
    ----------
    num_classes : int
        Number of output classification classes. Default is 10.
    stage_repeats : Sequence[int]
        Number of blocks in each stage. Default is (4, 8, 4).
    stage_out_channels : Sequence[int]
        Output channels for each stage [conv1, stage2, stage3, stage4, conv5].
        Default is (24, 116, 232, 464, 1024) for 1.0x width.
    norm : str
        Normalization type ('group', 'batch', 'none'). Default is 'group'.
    """

    num_classes: int = 10
    stage_repeats: Sequence[int] = (4, 8, 4)
    stage_out_channels: Sequence[int] = (24, 116, 232, 464, 1024)
    norm: str = "group"

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = True, **kwargs: Any
    ) -> jnp.ndarray:
        # Initial 3x3 Conv
        init_channels = self.stage_out_channels[0]
        x = nn.Conv(
            features=init_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = get_norm_layer(self.norm, init_channels, train=train)(x)
        x = nn.relu(x)

        # 3 Stages (Stages 2, 3, 4)
        for stage_idx, num_blocks in enumerate(self.stage_repeats):
            out_c = self.stage_out_channels[stage_idx + 1]
            # First block downsamples with stride=2
            x = ShuffleV2Block(out_channels=out_c, stride=2, norm=self.norm)(
                x, train=train
            )
            # Remaining blocks maintain resolution with stride=1
            for _ in range(1, num_blocks):
                x = ShuffleV2Block(out_channels=out_c, stride=1, norm=self.norm)(
                    x, train=train
                )

        # Conv5 (1x1 projection to high-dimensional embedding)
        conv5_channels = self.stage_out_channels[-1]
        x = nn.Conv(
            features=conv5_channels,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = get_norm_layer(self.norm, conv5_channels, train=train)(x)
        x = nn.relu(x)

        # Global Average Pooling and classification head
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=self.num_classes)(x)
        return x
