"""VGG architectures in Flax/JAX with Global Average Pooling."""

from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from models.vgg.configs import CFG_VGG


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
    norm : str
        Normalization type ('group', 'batch', 'none'). Default is 'group'.
    dropout_rate : float
        Dropout probability in classifier head. Default is 0.0.
    """

    config: Sequence[int | str]
    num_classes: int = 10
    norm: str = "group"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = True, **kwargs: Any
    ) -> jnp.ndarray:
        for item in self.config:
            if item == "M":
                if x.shape[1] > 1 and x.shape[2] > 1:
                    x = nn.max_pool(
                        x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
                    )
            else:
                x = nn.Conv(
                    features=item, kernel_size=(3, 3), padding="SAME", use_bias=False
                )(x)
                x = get_norm_layer(self.norm, item, train=train)(x)
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
    norm: str = "group"
    dropout_rate: float = 0.0


class VGG19(VGG):
    """Standard VGG-19 backbone with Global Average Pooling."""

    config: Sequence[int | str] = tuple(CFG_VGG["VGG19"])
    num_classes: int = 10
    norm: str = "group"
    dropout_rate: float = 0.0
