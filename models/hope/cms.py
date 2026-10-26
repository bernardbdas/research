"""Continuous Memory System (CMS) Module.

Evolves the Titans Neural Long-Term Memory and Google DeepMind's HOPE architecture
into an associative, continuum-scale memory module that operates in the forward pass
with adaptive forgetting and surprise-driven self-modification.

References
----------
1. Behrouz, A., et al. (2024). Titans: Learning to Memorize at Test Time. arXiv:2412.20231.
2. HOPE: Higher-Order Predictive Memory & Continuum Learning (Google DeepMind, 2025).
3. Ba, J., et al. (2016). Using Fast Weights to Attend to the Recent Past. NeurIPS.
"""

import flax.linen as nn
import jax.numpy as jnp


class ContinuousMemorySystem(nn.Module):
    """Continuous Memory System (CMS) with adaptive forgetting and surprise-based updates.

    Maintains a continuous memory matrix M in the forward pass that memorizes historical
    key-value associations and dynamically updates via decoupled gradient descent on the
    associative surprise signal:
        S_t = v_t - (k_t @ M_{t-1})
        M_t = alpha_t * M_{t-1} + eta_mem * (k_t^T @ S_t)
        read_t = q_t @ M_t

    Parameters
    ----------
    key_dim : int
        Dimension of key and query representations.
    val_dim : int
        Dimension of value representations.
    memory_lr : float
        Inner learning rate (eta_mem) for continuous forward-pass memory updates. Default is 0.05.
    use_adaptive_forgetting : bool
        Whether to learn an input-dependent retention/forgetting gate alpha_t. Default is True.
    """

    key_dim: int
    val_dim: int
    memory_lr: float = 0.05
    use_adaptive_forgetting: bool = True

    def setup(self) -> None:
        self.q_proj = nn.Dense(features=self.key_dim, use_bias=False)
        self.k_proj = nn.Dense(features=self.key_dim, use_bias=False)
        self.v_proj = nn.Dense(features=self.val_dim, use_bias=False)
        self.out_proj = nn.Dense(features=self.val_dim)

        if self.use_adaptive_forgetting:
            self.forget_gate = nn.Dense(features=1, use_bias=True)

        # Base persistent memory matrix initialization
        self.init_mem = self.param(
            "init_memory",
            nn.initializers.zeros,
            (self.key_dim, self.val_dim),
        )

    def __call__(
        self,
        x: jnp.ndarray,
        memory_state: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through Continuous Memory System.

        Parameters
        ----------
        x : jnp.ndarray
            Input representation of shape (B, D) or (B, T, D).
        memory_state : Optional[jnp.ndarray]
            Prior memory state matrix of shape (key_dim, val_dim) or (B, key_dim, val_dim).

        Returns
        -------
        output : jnp.ndarray
            Memory-augmented representation with same shape as x.
        updated_memory : jnp.ndarray
            Updated continuous memory state matrix.
        """
        # Handle 2D inputs (B, D) by expanding sequence dimension
        is_2d = x.ndim == 2
        if is_2d:
            x_seq = jnp.expand_dims(x, axis=1)  # (B, 1, D)
        else:
            x_seq = x  # (B, T, D)

        b_size, seq_len, _ = x_seq.shape

        q = self.q_proj(x_seq)  # (B, T, key_dim)
        k = self.k_proj(x_seq)  # (B, T, key_dim)
        v = self.v_proj(x_seq)  # (B, T, val_dim)

        if memory_state is None:
            # Broadcast base persistent memory across batch
            m_curr = jnp.broadcast_to(
                self.init_mem, (b_size, self.key_dim, self.val_dim)
            )
        elif memory_state.ndim == 2:
            m_curr = jnp.broadcast_to(
                memory_state, (b_size, self.key_dim, self.val_dim)
            )
        else:
            m_curr = memory_state

        reads = []
        for t in range(seq_len):
            q_t = q[:, t, :]  # (B, key_dim)
            k_t = k[:, t, :]  # (B, key_dim)
            v_t = v[:, t, :]  # (B, val_dim)

            # Compute associative prediction and surprise: S = v - k @ M
            # k_t: (B, 1, key_dim), m_curr: (B, key_dim, val_dim) -> pred: (B, val_dim)
            pred = jnp.squeeze(jnp.matmul(jnp.expand_dims(k_t, 1), m_curr), axis=1)
            surprise = v_t - pred  # (B, val_dim)

            # Compute adaptive forgetting factor alpha_t
            if self.use_adaptive_forgetting:
                alpha_t = nn.sigmoid(self.forget_gate(x_seq[:, t, :]))  # (B, 1)
                alpha_t = jnp.expand_dims(alpha_t, 2)  # (B, 1, 1)
            else:
                alpha_t = 0.95

            # Associative update: dM = k_t^T @ surprise
            # k_t: (B, key_dim, 1), surprise: (B, 1, val_dim) -> delta_m: (B, key_dim, val_dim)
            delta_m = jnp.matmul(jnp.expand_dims(k_t, 2), jnp.expand_dims(surprise, 1))

            # Continuous memory state transition
            m_curr = alpha_t * m_curr + self.memory_lr * delta_m

            # Read from updated memory: read_t = q_t @ M_curr
            read_t = jnp.squeeze(jnp.matmul(jnp.expand_dims(q_t, 1), m_curr), axis=1)
            reads.append(read_t)

        mem_out = jnp.stack(reads, axis=1)  # (B, T, val_dim)
        projected = self.out_proj(mem_out)

        # Residual combination with input
        output = x_seq + projected

        if is_2d:
            output = jnp.squeeze(output, axis=1)

        return output, m_curr
