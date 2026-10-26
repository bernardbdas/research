"""Vision-TTT: Test-Time Training Architecture for Vision Recognition.

Applies TTT-Linear layers where the hidden state is a neural network weight matrix
that self-trains via gradient descent at test time.

Reference
---------
Sun, Y., Li, X., et al. (2024).
Learning to (Learn at Test Time): RNNs with Expressive Hidden States.
arXiv:2407.04620.
"""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from models.ttt.layer import TTTLinearLayer


class TTTBlock(nn.Module):
    """TTT Block combining TTT-Linear layer, LayerNorm, and Gated MLP."""

    dim: int
    inner_lr: float = 0.05

    def setup(self) -> None:
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.ttt_layer = TTTLinearLayer(dim=self.dim, inner_lr=self.inner_lr)
        self.mlp_fc1 = nn.Dense(features=self.dim * 2)
        self.mlp_fc2 = nn.Dense(features=self.dim)

    def __call__(
        self,
        x: jnp.ndarray,
        test_time_update: bool = True,
    ) -> jnp.ndarray:
        # Branch 1: TTT-Linear layer with residual connection
        h = self.ttt_layer(self.norm1(x), test_time_update=test_time_update)
        x = x + h

        # Branch 2: Feed-forward network with residual connection
        m = nn.gelu(self.mlp_fc1(self.norm2(x)))
        x = x + self.mlp_fc2(m)
        return x


class VisionTTT(nn.Module):
    """Vision Test-Time Training (Vision-TTT) network.

    Parameters
    ----------
    num_classes : int
        Number of output target classes. Default is 10.
    patch_size : int
        Spatial size of each non-overlapping patch. Default is 4.
    embed_dim : int
        Dimensionality of patch tokens. Default is 64.
    depth : int
        Number of stacked TTT blocks. Default is 2.
    inner_lr : float
        Learning rate for inner test-time gradient steps. Default is 0.05.
    """

    num_classes: int = 10
    patch_size: int = 4
    embed_dim: int = 64
    depth: int = 2
    inner_lr: float = 0.05

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = True,
        test_time_update: bool = True,
        **kwargs: Any,
    ) -> jnp.ndarray:
        b, h, w, c = x.shape
        p = self.patch_size

        # 1. Patch projection
        x_patches = nn.Conv(
            features=self.embed_dim,
            kernel_size=(p, p),
            strides=(p, p),
            padding="VALID",
            name="patch_embed",
        )(x)
        num_patches = (h // p) * (w // p)
        tokens = x_patches.reshape((b, num_patches, self.embed_dim))

        # 2. Positional embeddings + LayerNorm
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches, self.embed_dim),
        )
        tokens = tokens + pos_embed
        tokens = nn.LayerNorm(name="norm_tokens")(tokens)

        # 3. Stacked TTT Blocks
        for i in range(self.depth):
            block = TTTBlock(
                dim=self.embed_dim,
                inner_lr=self.inner_lr,
                name=f"ttt_block_{i}",
            )
            tokens = block(tokens, test_time_update=test_time_update)

        # 4. Global representation pooling & classification head
        tokens = nn.LayerNorm(name="final_norm")(tokens)
        global_rep = jnp.mean(tokens, axis=1)
        logits = nn.Dense(features=self.num_classes, name="classifier")(global_rep)
        return logits
