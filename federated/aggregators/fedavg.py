"""Federated Averaging (FedAvg) algorithm.

Reference
---------
McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
Communication-efficient learning of deep networks from decentralized data.
AISTATS 2017.
"""

from typing import Any

import jax

from federated.aggregators.base import BaseAggregator
from federated.aggregators.registry import register_aggregator


@register_aggregator("fedavg", "fed_avg", "averaging")
class FedAvg(BaseAggregator):
    """Federated Averaging (FedAvg) algorithm (McMahan et al., 2017).

    Computes a sample-weighted (or uniform) linear combination of client parameters:
        w_{t+1} = sum_{k=1}^K p_k * w_k
    """

    def __init__(self, weighted: bool = True) -> None:
        super().__init__(name="FedAvg")
        self.weighted = weighted

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
            total_samples = float(sum(client_weights))
            norm_weights = (
                [w / total_samples for w in client_weights]
                if total_samples > 0
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
