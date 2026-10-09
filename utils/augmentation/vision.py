"""Vectorized vision data augmentation utilities."""

import numpy as np


def augment_cifar10_batch(
    images: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply vectorized random horizontal flip and 32x32 crop with 4px reflection padding.

    Parameters
    ----------
    images : np.ndarray
        Batch of images of shape (B, 32, 32, C).
    rng : Optional[np.random.Generator]
        NumPy random generator instance. If None, default generator is used.

    Returns
    -------
    np.ndarray
        Augmented batch of images with shape (B, 32, 32, C).
    """
    if rng is None:
        rng = np.random.default_rng()

    b, h, w, _ = images.shape

    # 1. Random horizontal flip (50% probability)
    flips = rng.random(b) > 0.5
    flipped = np.where(flips[:, None, None, None], images[:, :, ::-1, :], images)

    # 2. Reflection padding of 4 pixels on spatial dimensions
    padded = np.pad(flipped, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")

    # 3. Vectorized random crop back to original resolution (32x32)
    crop_y = rng.integers(0, 9, size=(b, 1, 1))
    crop_x = rng.integers(0, 9, size=(b, 1, 1))

    y_idx = crop_y + np.arange(h)[None, :, None]
    x_idx = crop_x + np.arange(w)[None, None, :]
    b_idx = np.arange(b)[:, None, None]

    return padded[b_idx, y_idx, x_idx, :]
