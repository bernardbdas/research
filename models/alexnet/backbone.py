"""AlexNet architecture in Flax/JAX with Global Average Pooling."""

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


class AlexNet(nn.Module):
    """Generic AlexNet backbone for Federated Learning.

    Features 5 convolutional layers followed by Global Average Pooling
    and multi-layer dense classification heads.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default is 10.
    norm : str
        Normalization type ('group', 'batch', 'none'). Default is 'group'.
    dropout_rate : float
        Dropout probability in classifier head. Default is 0.0.
    """

    num_classes: int = 10
    norm: str = "group"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = True, **kwargs: Any
    ) -> jnp.ndarray:
        # Conv 1
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = get_norm_layer(self.norm, 64, train=train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Conv 2
        x = nn.Conv(features=192, kernel_size=(3, 3), padding="SAME", use_bias=False)(x)
        x = get_norm_layer(self.norm, 192, train=train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Conv 3
        x = nn.Conv(features=384, kernel_size=(3, 3), padding="SAME", use_bias=False)(x)
        x = get_norm_layer(self.norm, 384, train=train)(x)
        x = nn.relu(x)

        # Conv 4
        x = nn.Conv(features=256, kernel_size=(3, 3), padding="SAME", use_bias=False)(x)
        x = get_norm_layer(self.norm, 256, train=train)(x)
        x = nn.relu(x)

        # Conv 5
        x = nn.Conv(features=256, kernel_size=(3, 3), padding="SAME", use_bias=False)(x)
        x = get_norm_layer(self.norm, 256, train=train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Global Average Pooling and classification head
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=4096)(x)
        x = nn.relu(x)
        if self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x
