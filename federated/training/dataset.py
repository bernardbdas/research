"""Partition extraction and array normalization for federated training."""

from typing import Any

import jax.numpy as jnp
import numpy as np


def extract_partition_arrays(
    part: Any,
    img_key: str,
    max_samples: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert a Flower client dataset partition into normalized JAX arrays.

    Parameters
    ----------
    part : Any
        Hugging Face / Flower dataset partition.
    img_key : str
        Column name for images ('img' or 'image').
    max_samples : int
        Maximum number of samples to extract.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Tuple of (images_array, labels_array).
    """
    sub = part.select(range(min(len(part), max_samples)))
    imgs = [
        np.expand_dims(np.array(im, dtype=np.float32) / 255.0, -1)
        if np.array(im).ndim == 2
        else np.array(im, dtype=np.float32) / 255.0
        for im in sub[img_key]
    ]
    xs = jnp.array(np.stack(imgs))
    ys = jnp.array(np.array(sub["label"], dtype=np.int32))
    return xs, ys


# Backward compatibility alias
_extract_partition_arrays = extract_partition_arrays
