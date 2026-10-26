"""Federated Muon (FedMuon) aggregation algorithm.

Applies the Muon (Momentum Orthogonalized by Newton-Schulz) optimizer as a
server-side federated optimization strategy with matrix orthogonalization.

References
----------
1. FedMuon: Federated Learning with Bias-corrected LMO-based Optimization (arXiv:2509.26337, 2025).
2. On Provable Benefits of Muon in Federated Learning (arXiv:2510.03866, 2025).
3. Jordan, K., et al. (2024). Muon: An optimizer for hidden layers in neural networks.
"""

from typing import Any

import jax
import jax.numpy as jnp

from federated.aggregators.base import BaseAggregator
from federated.aggregators.registry import register_aggregator


def zeropower_via_newtonschulz5(
    g: jnp.ndarray,
    steps: int = 5,
    eps: float = 1e-7,
) -> jnp.ndarray:
    """Newton-Schulz quintic iteration to approximate polar matrix orthogonalization.

    Approximates U @ V.T where G = U @ S @ V.T using 5th-order polynomial iterations.
    For tensors with ndim > 2 (e.g. convolution kernels), the tensor is reshaped to 2D
    across the channel dimension prior to orthogonalization.

    Parameters
    ----------
    g : jnp.ndarray
        Parameter displacement / gradient tensor of ndim >= 2.
    steps : int
        Number of Newton-Schulz iteration steps. Default is 5.
    eps : float
        Numerical stability epsilon.

    Returns
    -------
    jnp.ndarray
        Orthogonalized update matrix with identical shape as input.
    """
    orig_shape = g.shape
    if g.ndim > 2:
        g = g.reshape(-1, orig_shape[-1])

    d1, d2 = g.shape
    transposed = False
    if d1 > d2:
        g = g.T
        transposed = True

    # Scale spectral norm to <= 1
    norm = jnp.linalg.norm(g) + eps
    x = g / norm

    # Quintic Newton-Schulz polynomial coefficients
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        mat_a = x @ x.T
        mat_b = b * mat_a + c * (mat_a @ mat_a)
        x = a * x + mat_b @ x

    if transposed:
        x = x.T

    # Scale by aspect ratio to preserve parameter variance
    scale = jnp.sqrt(jnp.maximum(1.0, float(d1) / float(d2)))
    x = x * scale

    if len(orig_shape) > 2:
        x = x.reshape(orig_shape)
    return x


@register_aggregator("fedmuon", "fed_muon", "muon")
class FedMuon(BaseAggregator):
    """Federated Muon (FedMuon) server-side aggregation optimizer.

    Treats the average client displacement Delta_t = server_params - mean(client_params)
    as a pseudo-gradient, tracking momentum and applying Newton-Schulz matrix
    orthogonalization for multi-dimensional weight matrices and standard adaptive
    updates for 1D vectors/biases.

    Parameters
    ----------
    eta : float
        Server learning rate. Default is 0.05.
    momentum : float
        Server momentum factor beta1. Default is 0.9.
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
        momentum: float = 0.9,
        ns_steps: int = 5,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        adam_beta2: float = 0.99,
        adam_eps: float = 1e-6,
    ) -> None:
        super().__init__(name=f"FedMuon(eta={eta},ns={ns_steps})")
        self.eta = eta
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps

        self._m: Any | None = None
        self._v: Any | None = None
        self._step: int = 0

    def reset(self) -> None:
        """Reset internal round-to-round momentum buffers."""
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

        # 3. Server pseudo-gradient: delta = server_params - w_bar
        delta = jax.tree_util.tree_map(
            lambda w_srv, w_cli: w_srv - w_cli,
            server_params,
            w_bar,
        )

        # 4. Initialize state buffers if needed
        if self._m is None or self._v is None:
            self._m = jax.tree_util.tree_map(jnp.zeros_like, delta)
            self._v = jax.tree_util.tree_map(jnp.zeros_like, delta)

        self._step += 1
        t = self._step
        b1, b2 = self.momentum, self.adam_beta2
        eta, wd = self.eta, self.weight_decay
        ns_steps, nesterov = self.ns_steps, self.nesterov
        eps = self.adam_eps

        # 5. Update first momentum buffer: M_t = b1 * M_{t-1} + (1 - b1) * delta
        self._m = jax.tree_util.tree_map(
            lambda m_val, d_val: b1 * m_val + (1.0 - b1) * d_val,
            self._m,
            delta,
        )

        # 6. Update second moment for 1D non-matrix parameters: V_t = b2 * V_{t-1} + (1 - b2) * delta^2
        self._v = jax.tree_util.tree_map(
            lambda v_val, d_val: b2 * v_val + (1.0 - b2) * (d_val**2),
            self._v,
            delta,
        )

        # 7. Compute update direction per tensor leaf
        def _compute_leaf_update(
            w_val: jnp.ndarray,
            m_val: jnp.ndarray,
            v_val: jnp.ndarray,
            d_val: jnp.ndarray,
        ) -> jnp.ndarray:
            # Bias correction
            m_hat = m_val / (1.0 - (b1**t))

            # Apply Nesterov acceleration if configured
            if nesterov:
                grad_dir = b1 * m_hat + (1.0 - b1) * d_val
            else:
                grad_dir = m_hat

            if w_val.ndim >= 2:
                # 2D+ matrix/conv weight: apply Newton-Schulz orthogonalization
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
            self._m,
            self._v,
            delta,
        )
        return new_params
