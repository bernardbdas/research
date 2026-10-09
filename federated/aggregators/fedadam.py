"""Server-side Adaptive Moment Optimization (FedAdam / FedOpt).

Reference
---------
Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S., & McMahan, H. B. (2021).
Adaptive federated optimization.
ICLR 2021.
"""

from typing import Any

import jax
import jax.numpy as jnp

from federated.aggregators.base import BaseAggregator
from federated.aggregators.registry import register_aggregator


@register_aggregator("fedadam", "fed_adam", "adam")
class FedAdam(BaseAggregator):
    """Server-side Adaptive Moment Optimization (FedAdam / FedOpt) (Reddi et al., 2021).

    Treats the average client displacement Delta_t = server_params - mean(client_params)
    as a pseudo-gradient, applying Adam momentum and second-moment updates on the server.
    """

    def __init__(
        self,
        eta: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-4,
    ) -> None:
        super().__init__(name=f"FedAdam(eta={eta})")
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: Any | None = None
        self._v: Any | None = None
        self._step: int = 0

    def reset(self) -> None:
        self._m = None
        self._v = None
        self._step = 0

    def aggregate(
        self,
        server_params: Any,
        client_params: list[Any],
        client_weights: list[float] | None = None,
    ) -> Any:
        num_clients = len(client_params)
        if num_clients == 0:
            return server_params

        # Compute normalized client weights
        if client_weights is not None:
            total = float(sum(client_weights))
            norm_weights = (
                [w / total for w in client_weights]
                if total > 0
                else [1.0 / num_clients] * num_clients
            )
        else:
            norm_weights = [1.0 / num_clients] * num_clients

        # Average model across clients: w_bar = sum(p_k * w_k)
        w_bar = jax.tree_util.tree_map(
            lambda *leaves: sum(
                p * leaf for p, leaf in zip(norm_weights, leaves, strict=True)
            ),
            *client_params,
        )

        # Server pseudo-gradient: delta = server_params - w_bar
        delta = jax.tree_util.tree_map(
            lambda w_srv, w_cli: w_srv - w_cli,
            server_params,
            w_bar,
        )

        # Initialize momentum buffers on first round
        if self._m is None or self._v is None:
            self._m = jax.tree_util.tree_map(jnp.zeros_like, delta)
            self._v = jax.tree_util.tree_map(jnp.zeros_like, delta)

        self._step += 1
        t = self._step
        b1, b2, eta, eps = self.beta1, self.beta2, self.eta, self.eps

        # Update first and second moments
        self._m = jax.tree_util.tree_map(
            lambda m_val, d_val: b1 * m_val + (1.0 - b1) * d_val,
            self._m,
            delta,
        )
        self._v = jax.tree_util.tree_map(
            lambda v_val, d_val: b2 * v_val + (1.0 - b2) * (d_val**2),
            self._v,
            delta,
        )

        # Bias correction
        m_hat = jax.tree_util.tree_map(lambda m_val: m_val / (1.0 - (b1**t)), self._m)
        v_hat = jax.tree_util.tree_map(lambda v_val: v_val / (1.0 - (b2**t)), self._v)

        # Server model update: w_{t+1} = w_t - eta * m_hat / (sqrt(v_hat) + eps)
        new_params = jax.tree_util.tree_map(
            lambda w_val, m_h, v_h: w_val - eta * m_h / (jnp.sqrt(v_h) + eps),
            server_params,
            m_hat,
            v_hat,
        )
        return new_params
