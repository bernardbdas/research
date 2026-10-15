"""Federated Multi-scale Momentum Muon Optimizer (FedM3).

Integrates multi-scale momentum tracking (fast local drift tracking + slow global manifold
curvature) with quintic Newton-Schulz matrix polar orthogonalization for decentralized
federated learning.

References
----------
1. FedM3 Framework (Das, 2026): Decentralized Foundation Architecture with Continuous
   Memory System (CMS), HOPE-FL, and Federated Multi-scale Momentum Muon Optimization.
2. Takezawa, Y., et al. (2025). FedMuon: Federated Learning with Bias-corrected
   LMO-based Optimization. arXiv:2502.14815.
3. Jordan, K., et al. (2024). Muon: An optimizer for hidden layers in neural networks.
4. Behrouz, A., et al. (2024). Titans: Learning to Memorize at Test Time. arXiv:2412.20231.
"""

from typing import Any

import jax
import jax.numpy as jnp

from federated.aggregators.base import BaseAggregator
from federated.aggregators.fedmuon import zeropower_via_newtonschulz5
from federated.aggregators.registry import register_aggregator


@register_aggregator("fedm3", "fed_m3", "m3", "federated_m3")
class FedM3(BaseAggregator):
    """Federated Multi-scale Momentum Muon Optimizer (FedM3).

    Maintains dual-scale temporal momentum buffers to resolve the client drift vs. global
    curvature trade-off under heterogeneous Non-IID partitions:
    - Fast momentum buffer (beta_fast): tracks high-frequency client drift and local gradients.
    - Slow momentum buffer (beta_slow): captures low-frequency consensus manifold curvature.
    - Matrix Polar Orthogonalization: projects the multi-scale blended direction onto the
      orthogonal matrix manifold using 5th-order Newton-Schulz iterations.

    Parameters
    ----------
    eta : float
        Server learning rate. Default is 0.05.
    beta_fast : float
        Fast-scale momentum coefficient. Default is 0.8.
    beta_slow : float
        Slow-scale momentum coefficient. Default is 0.98.
    gamma : float
        Blending coefficient between fast and slow momentum (gamma * fast + (1 - gamma) * slow).
        Default is 0.6.
    ns_steps : int
        Number of quintic Newton-Schulz orthogonalization steps. Default is 5.
    nesterov : bool
        Whether to employ Nesterov momentum acceleration. Default is True.
    weight_decay : float
        Server-side weight decay coefficient. Default is 0.0.
    adam_beta2 : float
        Second-moment decay for 1D non-matrix parameters. Default is 0.99.
    adam_eps : float
        Numerical epsilon for 1D parameter normalization. Default is 1e-6.
    """

    def __init__(
        self,
        eta: float = 0.05,
        beta_fast: float = 0.8,
        beta_slow: float = 0.98,
        gamma: float = 0.6,
        ns_steps: int = 5,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        adam_beta2: float = 0.99,
        adam_eps: float = 1e-6,
    ) -> None:
        super().__init__(
            name=f"FedM3(eta={eta},fast={beta_fast},slow={beta_slow},gamma={gamma})"
        )
        self.eta = eta
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.gamma = gamma
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps

        self._m_fast: Any | None = None
        self._m_slow: Any | None = None
        self._v: Any | None = None
        self._step: int = 0

    def reset(self) -> None:
        """Reset internal round-to-round multi-scale momentum buffers."""
        self._m_fast = None
        self._m_slow = None
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

        # 1. Compute normalized client weights
        if client_weights is not None:
            total = float(sum(client_weights))
            norm_weights = (
                [w / total for w in client_weights]
                if total > 0
                else [1.0 / num_clients] * num_clients
            )
        else:
            norm_weights = [1.0 / num_clients] * num_clients

        # 2. Average client model weights: w_bar = sum(p_k * w_k)
        w_bar = jax.tree_util.tree_map(
            lambda *leaves: sum(
                p * leaf for p, leaf in zip(norm_weights, leaves, strict=True)
            ),
            *client_params,
        )

        # 3. Server pseudo-gradient displacement: delta = server_params - w_bar
        delta = jax.tree_util.tree_map(
            lambda w_srv, w_cli: w_srv - w_cli,
            server_params,
            w_bar,
        )

        # 4. Initialize multi-scale momentum buffers if needed
        if self._m_fast is None or self._m_slow is None or self._v is None:
            self._m_fast = jax.tree_util.tree_map(jnp.zeros_like, delta)
            self._m_slow = jax.tree_util.tree_map(jnp.zeros_like, delta)
            self._v = jax.tree_util.tree_map(jnp.zeros_like, delta)

        self._step += 1
        t = self._step
        b_fast, b_slow = self.beta_fast, self.beta_slow
        b2 = self.adam_beta2
        gamma = self.gamma
        eta, wd = self.eta, self.weight_decay
        ns_steps, nesterov = self.ns_steps, self.nesterov
        eps = self.adam_eps

        # 5. Update fast-scale momentum buffer: M_fast = b_fast * M_{t-1} + (1 - b_fast) * delta
        self._m_fast = jax.tree_util.tree_map(
            lambda m_val, d_val: b_fast * m_val + (1.0 - b_fast) * d_val,
            self._m_fast,
            delta,
        )

        # 6. Update slow-scale momentum buffer: M_slow = b_slow * M_{t-1} + (1 - b_slow) * delta
        self._m_slow = jax.tree_util.tree_map(
            lambda m_val, d_val: b_slow * m_val + (1.0 - b_slow) * d_val,
            self._m_slow,
            delta,
        )

        # 7. Update second moment for 1D non-matrix parameters: V_t = b2 * V_{t-1} + (1 - b2) * delta^2
        self._v = jax.tree_util.tree_map(
            lambda v_val, d_val: b2 * v_val + (1.0 - b2) * (d_val**2),
            self._v,
            delta,
        )

        # 8. Compute update direction per tensor leaf
        def _compute_leaf_update(
            w_val: jnp.ndarray,
            mf_val: jnp.ndarray,
            ms_val: jnp.ndarray,
            v_val: jnp.ndarray,
            d_val: jnp.ndarray,
        ) -> jnp.ndarray:
            # Dual bias corrections
            mf_hat = mf_val / (1.0 - (b_fast**t))
            ms_hat = ms_val / (1.0 - (b_slow**t))

            # Multi-scale blended direction
            m_blend = gamma * mf_hat + (1.0 - gamma) * ms_hat

            # Apply Nesterov acceleration if configured
            if nesterov:
                grad_dir = b_fast * m_blend + (1.0 - b_fast) * d_val
            else:
                grad_dir = m_blend

            if w_val.ndim >= 2:
                # 2D+ matrix/conv weight: apply Newton-Schulz polar orthogonalization
                update_dir = zeropower_via_newtonschulz5(grad_dir, steps=ns_steps)
            else:
                # 1D vector (bias, batch norm scale): apply coordinate-wise Adam update
                v_hat = v_val / (1.0 - (b2**t))
                update_dir = grad_dir / (jnp.sqrt(v_hat) + eps)

            # Apply optional weight decay: w_{t+1} = w_t - eta * (update_dir + wd * w_t)
            if wd > 0.0:
                update_dir = update_dir + wd * w_val

            return w_val - eta * update_dir

        new_params = jax.tree_util.tree_map(
            _compute_leaf_update,
            server_params,
            self._m_fast,
            self._m_slow,
            self._v,
            delta,
        )
        return new_params
