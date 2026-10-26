"""Pluggable Test-Time Memory Cartridges for Titans and Federated Personalization."""

import flax.linen as nn
import jax.numpy as jnp

from models.titans.memory import NeuralMemory


class MemoryCartridge(nn.Module):
    """Modular, pluggable Test-Time Training Cartridge.

    Holds client-specific persistent memory prefix tokens and a dedicated
    self-modifying associative memory bank that can be swapped, fine-tuned,
    or adapted on-the-fly at test time.

    Parameters
    ----------
    embed_dim : int
        Token representation dimensionality.
    num_persistent_tokens : int
        Number of learnable persistent memory tokens in this cartridge. Default is 4.
    memory_lr : float
        Inner learning rate for test-time memory self-modification. Default is 0.1.
    """

    embed_dim: int
    num_persistent_tokens: int = 4
    memory_lr: float = 0.1

    def setup(self) -> None:
        # Client/task persistent memory prefix tokens
        self.persistent_tokens = self.param(
            "persistent_tokens",
            nn.initializers.normal(stddev=0.02),
            (self.num_persistent_tokens, self.embed_dim),
        )
        self.q_proj = nn.Dense(features=self.embed_dim, use_bias=False)
        self.k_proj = nn.Dense(features=self.embed_dim, use_bias=False)
        self.v_proj = nn.Dense(features=self.embed_dim, use_bias=False)
        self.out_proj = nn.Dense(features=self.embed_dim)
        self.neural_mem = NeuralMemory(
            key_dim=self.embed_dim,
            val_dim=self.embed_dim,
            learning_rate=self.memory_lr,
        )

    def attach_prefix(self, x: jnp.ndarray) -> jnp.ndarray:
        """Prepend cartridge persistent memory tokens to an input sequence.

        Parameters
        ----------
        x : jnp.ndarray
            Input sequence of shape (B, T, embed_dim).

        Returns
        -------
        jnp.ndarray
            Prefixed sequence of shape (B, num_persistent_tokens + T, embed_dim).
        """
        b = x.shape[0]
        prefix = jnp.broadcast_to(
            self.persistent_tokens, (b, self.num_persistent_tokens, self.embed_dim)
        )
        return jnp.concatenate([prefix, x], axis=1)

    def __call__(
        self,
        x: jnp.ndarray,
        test_time_update: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the cartridge memory bank over the sequence with test-time surprise updates.

        Parameters
        ----------
        x : jnp.ndarray
            Sequence of shape (B, T, embed_dim).
        test_time_update : bool
            Whether memory self-modifies at test time.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            (adapted_features, final_cartridge_state).
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        retrieved, final_mem = self.neural_mem(
            q=q,
            k=k,
            v=v,
            test_time_update=test_time_update,
        )
        out = self.out_proj(retrieved)
        return out, final_mem
