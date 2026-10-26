"""Vision Transformer building blocks in Flax/JAX."""

import flax.linen as nn
import jax.numpy as jnp


class TransformerEncoderBlock(nn.Module):
    """Pre-LN Transformer Encoder Block with Multi-Head Attention and MLP."""

    embed_dim: int = 192
    num_heads: int = 3
    mlp_dim: int = 768
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Pre-LN Self-Attention
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(y, y)
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
        x = x + y

        # Pre-LN MLP
        y = nn.LayerNorm()(x)
        y = nn.Dense(features=self.mlp_dim)(y)
        y = nn.gelu(y)
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
        y = nn.Dense(features=self.embed_dim)(y)
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
        return x + y
