"""Federated Proximal (FedProx) algorithm.

Reference
---------
Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
Federated optimization in heterogeneous networks.
MLSys 2020.
"""

from typing import Any

import jax

from federated.aggregators.base import BaseAggregator
from federated.aggregators.registry import register_aggregator


@register_aggregator("fedprox", "fed_prox")
class FedProx(BaseAggregator):
    """Federated Proximal (FedProx) algorithm (Li et al., 2020).

    Adds a proximal regularization term to local client loss functions to constrain
    client drift under non-IID partitions, followed by weighted aggregation.
    """

    def __init__(self, mu: float = 0.01, weighted: bool = True) -> None:
        super().__init__(name=f"FedProx(mu={mu})")
        self.mu = float(mu)
        self.weighted = weighted

    @property
    def requires_proximal_loss(self) -> bool:
        return self.mu > 0.0

    @property
    def proximal_mu(self) -> float:
        return self.mu

    def aggregate(
        self,
        server_params: Any,
        client_params: list[Any],
        client_weights: list[float] | None = None,
    ) -> Any:
        num_clients = len(client_params)
        if num_clients == 0:
            return server_params

        if self.weighted and client_weights is not None:
            total = float(sum(client_weights))
            norm_weights = (
                [w / total for w in client_weights]
                if total > 0
                else [1.0 / num_clients] * num_clients
            )
        else:
            norm_weights = [1.0 / num_clients] * num_clients

        return jax.tree_util.tree_map(
            lambda *leaves: sum(
                p * leaf for p, leaf in zip(norm_weights, leaves, strict=True)
            ),
            *client_params,
        )
