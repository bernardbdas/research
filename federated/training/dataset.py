"""Partition extraction and array normalization for federated training."""

from typing import Any

import jax.numpy as jnp
import numpy as np


def extract_partition_arrays(
    part: Any,
    img_key: str | None = None,
    max_samples: int = 1000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert a Flower client dataset partition into normalized JAX arrays.

    Parameters
    ----------
    part : Any
        Hugging Face / Flower dataset partition.
    img_key : Optional[str]
        Column name for images ('img', 'image', etc.). If None or not in partition,
        auto-detected from partition column names.
    max_samples : int
        Maximum number of samples to extract. Default is 1000.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Tuple of (images_array, labels_array).
    """
    col_names = getattr(part, "column_names", [])

    if img_key is None or (col_names and img_key not in col_names):
        for candidate in ("image", "img", "images", "pixel_values"):
            if candidate in col_names:
                img_key = candidate
                break
        else:
            if "image" in col_names:
                img_key = "image"
            elif "img" in col_names:
                img_key = "img"
            elif col_names:
                img_key = col_names[0]
            else:
                img_key = "image"

    label_key = "label"
    if col_names and label_key not in col_names:
        for candidate in ("labels", "fine_label", "coarse_label", "target"):
            if candidate in col_names:
                label_key = candidate
                break

    sub = part.select(range(min(len(part), max_samples)))
    raw_images = sub[img_key]
    imgs = [
        np.expand_dims(np.array(im, dtype=np.float32) / 255.0, -1)
        if np.array(im).ndim == 2
        else np.array(im, dtype=np.float32) / 255.0
        for im in raw_images
    ]
    xs = jnp.array(np.stack(imgs))
    ys = jnp.array(np.array(sub[label_key], dtype=np.int32))
    return xs, ys


# Backward compatibility alias
_extract_partition_arrays = extract_partition_arrays
