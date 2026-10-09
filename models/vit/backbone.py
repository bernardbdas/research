"""Vision Transformer (ViT) architecture in Flax/JAX."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from models.vit.blocks import TransformerEncoderBlock


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) backbone for Federated Learning.

    Divides images into non-overlapping spatial patches, maps them via a linear
    projection to token embeddings, prepends a learnable CLS token, adds 1D
    learnable position embeddings, and processes through multiple Transformer
    Encoder blocks.

    Parameters
    ----------
    patch_size : int
        Size of square image patches (P). Default is 4.
    embed_dim : int
        Token embedding dimensionality (D). Default is 192.
    depth : int
        Number of Transformer encoder layers (L). Default is 6.
    num_heads : int
        Number of attention heads (H). Default is 3.
    mlp_dim : int
        Hidden dimension of MLP feedforward layers. Default is 768.
    num_classes : int
        Number of output classification classes. Default is 10.
    dropout_rate : float
        Dropout probability. Default is 0.0.
    """

    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_dim: int = 768
    num_classes: int = 10
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = True, **kwargs: Any
    ) -> jnp.ndarray:
        b, h, w, _ = x.shape

        # Patch projection via strided convolution
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=True,
        )(x)
        num_patches = (h // self.patch_size) * (w // self.patch_size)
        x = x.reshape((b, num_patches, self.embed_dim))

        # Learnable CLS token
        cls_token = self.param("cls", nn.initializers.zeros, (1, 1, self.embed_dim))
        cls_tokens = jnp.broadcast_to(cls_token, (b, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        # Learnable Position Embeddings
        pos_embed = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches + 1, self.embed_dim),
        )
        x = x + pos_embed

        if self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        # Transformer Encoder Blocks
        for _ in range(self.depth):
            x = TransformerEncoderBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        # Representation head over [CLS] token
        x = nn.LayerNorm()(x[:, 0])
        x = nn.Dense(features=self.num_classes)(x)
        return x


class ViTTiny(VisionTransformer):
    """ViT-Tiny backbone (embed_dim=192, depth=6, num_heads=3)."""

    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_dim: int = 768


class ViTSmall(VisionTransformer):
    """ViT-Small backbone (embed_dim=384, depth=8, num_heads=6)."""

    patch_size: int = 4
    embed_dim: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_dim: int = 1536
