"""Generic Simple CNN backbone for Federated Learning."""

import flax.linen as nn
import jax.numpy as jnp


class SimpleCNN(nn.Module):
    """Compact, data-agnostic CNN backbone for Federated Learning.

    Uses Global Average Pooling across spatial dimensions, allowing it
    to operate on inputs of arbitrary image resolutions and channel counts.

    Parameters
    ----------
    num_classes : int
        Number of output classification classes. Default is 10.
    dropout_rate : float
        Dropout probability. Default is 0.0.
    """

    num_classes: int = 10
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Block 1
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 2
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 3
        x = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Global Average Pooling (resolution agnostic)
        x = jnp.mean(x, axis=(1, 2))

        # Classifier head
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        if self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        x = nn.Dense(features=self.num_classes)(x)
        return x
