"""VGG architectures in Flax/JAX with Global Average Pooling."""

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

# VGG Configurations
# 'M' represents a MaxPool layer
CFG_VGG = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    """Generic, data-agnostic VGG backbone for Federated Learning.

    Uses Global Average Pooling before the classifier head to support
    arbitrary input spatial dimensions while keeping model communication size
    efficient.

    Parameters
    ----------
    config : Sequence[int | str]
        List of layer channel sizes or 'M' for max pooling.
    num_classes : int
        Number of output classes. Default is 10.
    use_batch_norm : bool
        Whether to include Batch Normalization after convolutions. Default is True.
    dropout_rate : float
        Dropout probability in classifier head. Default is 0.0.
    """

    config: Sequence[int | str]
    num_classes: int = 10
    use_batch_norm: bool = True
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        for item in self.config:
            if item == "M":
                x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            else:
                x = nn.Conv(features=item, kernel_size=(3, 3), padding="SAME")(x)
                if self.use_batch_norm:
                    x = nn.BatchNorm(use_running_average=not train)(x)
                x = nn.relu(x)

        # Global Average Pooling (spatial dimension reduction)
        x = jnp.mean(x, axis=(1, 2))

        # Classifier head
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        if self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        x = nn.Dense(features=self.num_classes)(x)
        return x


class VGG16(VGG):
    """Standard VGG-16 backbone with Global Average Pooling."""

    config: Sequence[int | str] = tuple(CFG_VGG["VGG16"])
    num_classes: int = 10
    use_batch_norm: bool = True
    dropout_rate: float = 0.0
