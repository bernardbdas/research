"""Parameter calculation and inspection utilities for Flax neural networks."""

from typing import Any

import jax


def count_parameters(variables: dict[str, Any]) -> int:
    """Calculate the total number of trainable parameters in a Flax parameter PyTree."""
    params = variables.get("params", variables)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
