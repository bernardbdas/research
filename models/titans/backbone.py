"""Vision Titans: Self-Modifying Titans Architecture for Vision Recognition.

Combines Persistent Memory, Short-Term Multi-Head Attention, and Long-Term
Surprise-based Neural Memory for test-time adaptation.

Reference
---------
Behrouz, A., Peebles, W., Zhang, K., et al. (2024).
Titans: Learning to Memorize at Test Time. Google Research.
arXiv:2412.20231.
"""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from models.titans.cartridges import MemoryCartridge
from models.titans.memory import NeuralMemory


class TitansBlock(nn.Module):
    """Core Titans Block integrating Short-Term Attention and Long-Term Memory.

    Fuses:
    1. Short-Term Memory: Multi-Head Self-Attention.
    2. Long-Term Memory: NeuralMemory with test-time surprise self-modification.
    3. Learned Adaptive Gating to balance short- and long-term representations.
    4. Gated Feed-Forward Network.

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads. Default is 4.
    memory_lr : float
        Inner learning rate for memory updates. Default is 0.05.
    """

    embed_dim: int
    num_heads: int = 4
    memory_lr: float = 0.05

    def setup(self) -> None:
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm_mlp = nn.LayerNorm()

        # 1. Short-Term Attention
        self.attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
        )

        # 2. Long-Term Memory projections & module
        self.q_mem = nn.Dense(features=self.embed_dim, use_bias=False)
        self.k_mem = nn.Dense(features=self.embed_dim, use_bias=False)
        self.v_mem = nn.Dense(features=self.embed_dim, use_bias=False)
        self.neural_mem = NeuralMemory(
            key_dim=self.embed_dim,
            val_dim=self.embed_dim,
            learning_rate=self.memory_lr,
        )

        # 3. Gating unit between short-term attention and neural memory
        self.gate_dense = nn.Dense(features=self.embed_dim)

        # 4. Feed-Forward Network
        self.mlp_fc1 = nn.Dense(features=self.embed_dim * 2)
        self.mlp_fc2 = nn.Dense(features=self.embed_dim)

    def __call__(
        self,
        x: jnp.ndarray,
        test_time_update: bool = True,
    ) -> jnp.ndarray:
        # Branch 1: Short-term Attention
        normed1 = self.norm1(x)
        attn_out = self.attn(normed1, normed1)

        # Branch 2: Long-term Neural Memory
        normed2 = self.norm2(x)
        qm = self.q_mem(normed2)
        km = self.k_mem(normed2)
        vm = self.v_mem(normed2)

        mem_out, _ = self.neural_mem(
            q=qm,
            k=km,
            v=vm,
            test_time_update=test_time_update,
        )

        # Gated fusion: g in (0, 1)
        gate = nn.sigmoid(self.gate_dense(x))
        fused = gate * attn_out + (1.0 - gate) * mem_out
        x = x + fused

        # Feed-Forward Network with residual connection
        mlp_in = self.norm_mlp(x)
        h = nn.gelu(self.mlp_fc1(mlp_in))
        mlp_out = self.mlp_fc2(h)
        x = x + mlp_out
        return x


class VisionTitans(nn.Module):
    """Vision Titans architecture for test-time adaptive classification.

    Processes images via patch tokenization, adds persistent memory tokens,
    and applies Titans blocks with self-modifying neural memory.

    Parameters
    ----------
    num_classes : int
        Number of output target classes. Default is 10.
    patch_size : int
        Spatial size of each non-overlapping patch. Default is 4.
    embed_dim : int
        Dimensionality of patch and memory embeddings. Default is 64.
    depth : int
        Number of stacked Titans blocks. Default is 2.
    num_heads : int
        Number of attention heads per block. Default is 4.
    num_persistent_tokens : int
        Number of learnable persistent memory tokens. Default is 4.
    memory_lr : float
        Inner learning rate for test-time surprise updates. Default is 0.05.
    use_cartridge : bool
        Whether to attach a pluggable MemoryCartridge. Default is True.
    """

    num_classes: int = 10
    patch_size: int = 4
    embed_dim: int = 64
    depth: int = 2
    num_heads: int = 4
    num_persistent_tokens: int = 4
    memory_lr: float = 0.05
    use_cartridge: bool = True

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

        # 1. Patch projection: extract PxP non-overlapping patches
        # Conv with kernel=P, stride=P acts as linear projection of patches
        x_patches = nn.Conv(
            features=self.embed_dim,
            kernel_size=(p, p),
            strides=(p, p),
            padding="VALID",
            name="patch_embed",
        )(x)
        # Flatten spatial grid into sequence of tokens: (B, num_patches, embed_dim)
        num_patches = (h // p) * (w // p)
        tokens = x_patches.reshape((b, num_patches, self.embed_dim))

        # 2. Add learnable positional embeddings + LayerNorm
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches, self.embed_dim),
        )
        tokens = tokens + pos_embed
        tokens = nn.LayerNorm(name="norm_tokens")(tokens)

        # 3. Attach Persistent Memory Cartridge if enabled
        if self.use_cartridge:
            cartridge = MemoryCartridge(
                embed_dim=self.embed_dim,
                num_persistent_tokens=self.num_persistent_tokens,
                memory_lr=self.memory_lr,
                name="cartridge",
            )
            tokens = cartridge.attach_prefix(tokens)

        # 4. Process through stacked Titans Blocks
        for i in range(self.depth):
            block = TitansBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                memory_lr=self.memory_lr,
                name=f"titans_block_{i}",
            )
            tokens = block(tokens, test_time_update=test_time_update)

        # 5. Global representation pooling & classification head
        tokens = nn.LayerNorm(name="final_norm")(tokens)
        global_rep = jnp.mean(tokens, axis=1)  # (B, embed_dim)
        logits = nn.Dense(features=self.num_classes, name="classifier")(global_rep)
        return logits
