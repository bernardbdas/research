"""Visual feature extractor for continual and transfer learning."""

import jax
import jax.numpy as jnp
import numpy as np

from models import get_model


class FlaxFeatureExtractor:
    """Extract semantic visual embeddings using a pure Flax backbone."""

    def __init__(
        self, architecture: str = "simple_cnn", embed_dim: int = 512, seed: int = 42
    ):
        self.embed_dim = embed_dim
        self.model = get_model(architecture, num_classes=embed_dim)
        key = jax.random.PRNGKey(seed)
        self.params = self.model.init(key, jnp.ones((1, 32, 32, 3)), train=False)

    def extract_features(self, images: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Extract continuous feature representations for a batch of images."""
        features = []
        for i in range(0, len(images), batch_size):
            bx = jnp.array(images[i : i + batch_size])
            out = self.model.apply(self.params, bx, train=False)
            features.append(np.array(out))
        return np.concatenate(features, axis=0)
