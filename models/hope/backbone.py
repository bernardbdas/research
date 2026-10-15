"""HOPE-FL: Federated Higher-Order Predictive Memory Architecture.

The natural evolution of Google DeepMind's HOPE architecture and Titans into
federated learning, integrating Continuous Memory Systems (CMS) with efficient
vision feature representations.

References
----------
1. Das, B. B. (2026). FedM3: Federated Multi-scale Momentum Muon Optimization with
   Continuous Memory Systems and HOPE-FL.
2. Behrouz, A., et al. (2024). Titans: Learning to Memorize at Test Time. arXiv:2412.20231.
3. HOPE: Higher-Order Predictive Memory & Continuum Learning (Google DeepMind, 2025).
"""

import flax.linen as nn
import jax.numpy as jnp

from models.hope.cms import ContinuousMemorySystem


class HOPEFLNet(nn.Module):
    """HOPE-FL Lightweight Architecture with Continuous Memory System.

    Designed for fast, reproducible execution on Apple Silicon (M1/M2/M3) and scalable
    to large-scale multi-GPU clusters. Fuses a convolutional feature projection trunk
    with a Continuous Memory System (CMS) module for test-time associative memorization.

    Parameters
    ----------
    num_classes : int
        Number of target classification categories. Default is 10.
    embed_dim : int
        Intermediate representation dimensionality. Default is 64.
    memory_lr : float
        Inner continuous memory update rate. Default is 0.05.
    use_adaptive_forgetting : bool
        Whether to enable dynamic input-dependent memory forgetting. Default is True.
    """

    num_classes: int = 10
    embed_dim: int = 64
    memory_lr: float = 0.05
    use_adaptive_forgetting: bool = True

    def setup(self) -> None:
        # 1. Lightweight Vision Feature Trunk
        self.conv1 = nn.Conv(
            features=16, kernel_size=(3, 3), strides=(2, 2), padding="SAME"
        )
        self.conv2 = nn.Conv(
            features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME"
        )
        self.proj = nn.Dense(features=self.embed_dim)

        # 2. Continuous Memory System (CMS)
        self.cms = ContinuousMemorySystem(
            key_dim=self.embed_dim,
            val_dim=self.embed_dim,
            memory_lr=self.memory_lr,
            use_adaptive_forgetting=self.use_adaptive_forgetting,
        )

        # 3. LayerNorm and Classification Head
        self.norm = nn.LayerNorm()
        self.classifier = nn.Dense(features=self.num_classes)

    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = True,
        memory_state: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Forward pass through HOPE-FL architecture.

        Parameters
        ----------
        x : jnp.ndarray
            Input image batch of shape (B, H, W, C).
        train : bool
            Training mode flag. Default is True.
        memory_state : Optional[jnp.ndarray]
            Optional continuous memory state matrix.

        Returns
        -------
        jnp.ndarray
            Logits of shape (B, num_classes).
        """
        # Ensure 4D (B, H, W, C)
        if x.ndim == 3:
            x = jnp.expand_dims(x, axis=-1)

        # Vision trunk
        h = nn.relu(self.conv1(x))
        h = nn.relu(self.conv2(h))
        h = h.reshape((h.shape[0], -1))
        h = self.proj(h)

        # Continuous Memory System (CMS) associative processing
        h_mem, _ = self.cms(h, memory_state=memory_state)

        # Classification Head
        h_norm = self.norm(h_mem)
        logits = self.classifier(h_norm)
        return logits
