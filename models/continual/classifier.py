"""Dynamic incremental classifier head for class-incremental learning."""

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


class IncrementalClassifier(nn.Module):
    """Flax module parameterized by output class count."""

    num_classes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(features=self.num_classes, name="fc")(x)
        return x


def expand_head_parameters(
    old_params: dict[str, Any],
    new_num_classes: int,
    key: jax.Array,
) -> dict[str, Any]:
    """Dynamically expand dense classification layer weight and bias matrices."""
    has_params_wrapper = "params" in old_params
    raw_params = old_params["params"] if has_params_wrapper else old_params
    old_w = raw_params["fc"]["kernel"]
    old_b = raw_params["fc"]["bias"]
    old_classes = old_w.shape[1]
    added_classes = new_num_classes - old_classes

    if added_classes <= 0:
        return old_params

    embed_dim = old_w.shape[0]
    k_w, k_b = jax.random.split(key)
    new_w_cols = jax.random.normal(k_w, (embed_dim, added_classes)) * (
        1.0 / jnp.sqrt(embed_dim)
    )
    new_b_cols = jnp.zeros((added_classes,))

    expanded_w = jnp.concatenate([old_w, new_w_cols], axis=1)
    expanded_b = jnp.concatenate([old_b, new_b_cols], axis=0)

    res = {"fc": {"kernel": expanded_w, "bias": expanded_b}}
    return {"params": res} if has_params_wrapper else res
