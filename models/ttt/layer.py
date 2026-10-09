"""TTT-Linear: Test-Time Training layer with gradient-based inner loop.

Reference
---------
Sun, Y., Li, X., et al. (2024).
Learning to (Learn at Test Time): RNNs with Expressive Hidden States.
arXiv:2407.04620.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


class TTTLinearLayer(nn.Module):
    """TTT-Linear layer where the hidden state is a model weight matrix.

    Updates its internal weight matrix W_t via an inner-loop gradient descent
    step on the self-supervised reconstruction objective:
        ell(W; x_t) = 0.5 * ||W @ k_t - v_t||^2

    Parameters
    ----------
    dim : int
        Feature dimensionality.
    inner_lr : float
        Learning rate for the inner test-time gradient step. Default is 0.1.
    """

    dim: int
    inner_lr: float = 0.05

    def setup(self) -> None:
        self.q_proj = nn.Dense(features=self.dim, use_bias=False)
        self.k_proj = nn.Dense(features=self.dim, use_bias=False)
        self.v_proj = nn.Dense(features=self.dim, use_bias=False)
        self.out_proj = nn.Dense(features=self.dim)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        test_time_update: bool = True,
    ) -> jnp.ndarray:
        """Forward pass over sequence with causal test-time gradient steps.

        Parameters
        ----------
        x : jnp.ndarray
            Input sequence of shape (B, T, dim).
        test_time_update : bool
            Whether to update the inner hidden weight matrix W across sequence steps.

        Returns
        -------
        jnp.ndarray
            Output sequence of shape (B, T, dim).
        """
        b, t, d = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Base initial weight matrix W_0 (learned parameter shared across tasks)
        w_base = self.param(
            "w_base",
            nn.initializers.normal(stddev=0.02),
            (d, d),
        )
        w_curr = jnp.broadcast_to(w_base, (b, d, d))

        scale = 1.0 / jnp.sqrt(d)

        def _update_w(
            operands: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> jnp.ndarray:
            w_c, k_step, v_step = operands
            # Normalize key vector for stable inner loop optimization (Sun et al., 2024)
            k_norm = k_step / (jnp.linalg.norm(k_step, axis=-1, keepdims=True) + 1e-5)
            # Inner loop reconstruction: v_hat = k_norm @ W
            v_hat = jnp.squeeze(jnp.matmul(k_norm[:, None, :], w_c), axis=1)
            # Analytical gradient d(0.5 * ||k_norm @ W - v||^2)/dW = k_norm^T @ (k_norm @ W - v)
            grad_w = jnp.matmul(
                k_norm[:, :, None], (v_hat - v_step)[:, None, :]
            )  # (B, D, D)
            # Inner gradient descent update: W = W - inner_lr * grad_W
            return w_c - self.inner_lr * grad_w

        def _keep_w(
            operands: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> jnp.ndarray:
            return operands[0]

        out_list = []
        for i in range(t):
            qi = q[:, i, :]  # (B, D)
            ki = k[:, i, :]  # (B, D)
            vi = v[:, i, :]  # (B, D)

            # Output forward transform: y_i = (q_i / sqrt(D)) @ W_curr
            yi = jnp.squeeze(jnp.matmul((qi * scale)[:, None, :], w_curr), axis=1)
            out_list.append(yi)

            w_curr = jax.lax.cond(
                test_time_update,
                _update_w,
                _keep_w,
                (w_curr, ki, vi),
            )

        outputs = jnp.stack(out_list, axis=1)  # (B, T, D)
        return self.out_proj(outputs)
