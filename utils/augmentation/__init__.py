"""Data augmentation, normalization, and optimization scheduling utilities."""

from utils.augmentation.normalization import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    normalize_cifar10,
)
from utils.augmentation.schedules import (
    create_cosine_schedule,
)
from utils.augmentation.vision import (
    augment_cifar10_batch,
)

__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "augment_cifar10_batch",
    "create_cosine_schedule",
    "normalize_cifar10",
]
