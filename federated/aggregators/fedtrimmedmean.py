"""Coordinate-wise Trimmed Mean aggregation (FedTrimmedMean).

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


@register_aggregator("fedtrimmedmean", "fed_trimmed_mean", "trimmed_mean")
class FedTrimmedMean(BaseAggregator):
    """Coordinate-wise Trimmed Mean aggregation (Yin et al., 2018).

    Discards the smallest and largest beta fraction of values along each parameter
    coordinate before computing the average.
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__(name=f"FedTrimmedMean(beta={beta})")
        if not (0.0 <= beta < 0.5):
            raise ValueError(f"beta must be in [0.0, 0.5), got {beta}")
        self.beta = beta

    def aggregate(
        self,
        server_params: Any,
        client_params: list[Any],
        client_weights: list[float] | None = None,
    ) -> Any:
        if not client_params:
            return server_params

        num_clients = len(client_params)
        trim_count = int(self.beta * num_clients)

        def _trim_and_mean(*leaves: Any) -> Any:
            stacked = jnp.stack(leaves, axis=0)
            if trim_count == 0 or 2 * trim_count >= num_clients:
                return jnp.mean(stacked, axis=0)
            sorted_stacked = jnp.sort(stacked, axis=0)
            trimmed = sorted_stacked[trim_count : num_clients - trim_count]
            return jnp.mean(trimmed, axis=0)

        return jax.tree_util.tree_map(_trim_and_mean, *client_params)
