"""Coordinate-wise Median aggregation (FedMedian).

Reference
---------
Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018).
Byzantine-robust distributed learning: Towards optimal statistical rates.
ICML 2018.
"""

from typing import Any

import jax
import jax.numpy as jnp

from federated.aggregators.base import BaseAggregator
from federated.aggregators.registry import register_aggregator


@register_aggregator("fedmedian", "fed_median", "median")
class FedMedian(BaseAggregator):
    """Coordinate-wise Median aggregation (Yin et al., 2018).

    Takes the element-wise median across client updates for each model parameter,
    providing robust protection against outlier or poisoned client models.
    """

    def __init__(self) -> None:
        super().__init__(name="FedMedian")

    def aggregate(
        self,
        server_params: Any,
        client_params: list[Any],
        client_weights: list[float] | None = None,
    ) -> Any:
        if not client_params:
            return server_params

        return jax.tree_util.tree_map(
            lambda *leaves: jnp.median(jnp.stack(leaves, axis=0), axis=0),
            *client_params,
        )
