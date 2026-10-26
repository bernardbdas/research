"""Normalization constants and transformation functions."""

import numpy as np

# Canonical per-channel mean and standard deviation for CIFAR-10
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def normalize_cifar10(images: np.ndarray) -> np.ndarray:
    """Standardize images by CIFAR-10 per-channel mean and standard deviation.

    Parameters
    ----------
    images : np.ndarray
        Array of images with values in [0.0, 1.0] and shape (..., H, W, 3).

    Returns
    -------
    np.ndarray
        Normalized images.
    """
    return (images - CIFAR10_MEAN) / CIFAR10_STD
