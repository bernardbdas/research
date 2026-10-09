"""Neural Long-Term Memory module with Surprise-based Test-Time updates for Titans.

Reference
---------
Behrouz, A., Peebles, W., Zhang, K., et al. (2024).
Titans: Learning to Memorize at Test Time. Google Research.
arXiv:2412.20231.
"""

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


class NeuralMemory(nn.Module):
    """Associative Neural Long-Term Memory with Surprise Metric and Adaptive Forgetting.

    Learns to dynamically self-modify at test time based on the surprise signal
    derived from associative reconstruction loss:
        L_mem(M_t) = 0.5 * ||M_{t-1}(k_t) - v_t||^2

    Parameters
    ----------
    key_dim : int
        Dimension of memory key representations.
    val_dim : int
        Dimension of memory value representations.
    momentum : float
        Momentum factor for accumulating surprise gradients. Default is 0.9.
    learning_rate : float
        Inner learning rate (step size) for memory self-modification. Default is 0.1.
    adaptive_forgetting : bool
        Whether to learn an input-dependent forgetting gate alpha_t. Default is True.
    """

    key_dim: int
    val_dim: int
    momentum: float = 0.9
    learning_rate: float = 0.05
    adaptive_forgetting: bool = True

    @nn.compact
    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        init_memory: jnp.ndarray | None = None,
        test_time_update: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Process sequence through neural memory with sequential surprise updates.

        Parameters
        ----------
        q : jnp.ndarray
            Queries of shape (B, T, key_dim).
        k : jnp.ndarray
            Keys of shape (B, T, key_dim).
        v : jnp.ndarray
            Values of shape (B, T, val_dim).
        init_memory : Optional[jnp.ndarray]
            Initial associative memory state of shape (B, key_dim, val_dim).
        test_time_update : bool
            Whether to dynamically update the memory matrix across sequence steps.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            (retrieved_values of shape (B, T, val_dim), final_memory_state).
        """
        b, t, dk = k.shape
        dv = v.shape[-1]
        scale = 1.0 / jnp.sqrt(dk)

        # Learned projections for adaptive forgetting gate if enabled
        if self.adaptive_forgetting:
            alpha_dense = nn.Dense(features=1, use_bias=True, name="forget_gate")

        # Initialize base memory state if not provided
        if init_memory is None:
            m_init = self.param(
                "m_base",
                nn.initializers.zeros,
                (dk, dv),
            )
            # Broadcast to batch size
            m_curr = jnp.broadcast_to(m_init, (b, dk, dv))
        else:
            m_curr = init_memory

        mom_curr = jnp.zeros_like(m_curr)
        retrieved_list = []

        def _update_mem(
            operands: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            m_c, mom_c, k_step, v_step, a_step = operands
            k_scaled = k_step * scale
            # 2. Compute surprise metric: reconstruction error on current key-value pair
            v_hat = jnp.squeeze(jnp.matmul(k_scaled[:, None, :], m_c), axis=1)
            # Error: Delta = v_hat - vi
            delta = v_hat - v_step  # (B, val_dim)

            # 3. Surprise gradient: S = ki^T @ delta
            surprise_grad = jnp.matmul(k_scaled[:, :, None], delta[:, None, :])

            # 4. Momentum accumulation
            mom_updated = self.momentum * mom_c + surprise_grad

            # 5. Memory self-modification step: M = (1 - alpha) * M - lr * momentum
            m_updated = (1.0 - a_step) * m_c - self.learning_rate * mom_updated
            return m_updated, mom_updated

        def _keep_mem(
            operands: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            return operands[0], operands[1]

        # Iterate over sequence tokens (causal associative memory update)
        for i in range(t):
            qi = q[:, i, :]  # (B, key_dim)
            ki = k[:, i, :]  # (B, key_dim)
            vi = v[:, i, :]  # (B, val_dim)

            # 1. Retrieve value from current memory state: y_i = (q_i / sqrt(dk)) @ M_curr
            yi = jnp.squeeze(jnp.matmul((qi * scale)[:, None, :], m_curr), axis=1)
            retrieved_list.append(yi)

            # Adaptive forgetting factor alpha in (0, 1)
            if self.adaptive_forgetting:
                alpha = nn.sigmoid(alpha_dense(ki))  # (B, 1)
                alpha = alpha[:, :, None]  # (B, 1, 1)
            else:
                alpha = 0.05

            m_curr, mom_curr = jax.lax.cond(
                test_time_update,
                _update_mem,
                _keep_mem,
                (m_curr, mom_curr, ki, vi, alpha),
            )

        retrieved = jnp.stack(retrieved_list, axis=1)  # (B, T, val_dim)
        return retrieved, m_curr
