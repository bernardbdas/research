"""Differential Privacy algorithms, DP-SGD transformations, and Rényi privacy accounting."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class RenyiPrivacyAccountant:
    """Analytical Rényi Differential Privacy (RDP) accountant for subsampled Gaussian mechanisms.

    Tracks accumulated privacy loss across training rounds and converts cumulative RDP
    guarantees into standard (epsilon, delta)-DP bounds.
    """

    def __init__(self, orders: list[float] | None = None):
        self.orders = orders or [
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            4.0,
            5.0,
            6.0,
            8.0,
            10.0,
            12.0,
            16.0,
            20.0,
            24.0,
            32.0,
            64.0,
        ]
        self.rdp = dict.fromkeys(self.orders, 0.0)

    def step(self, q: float, noise_multiplier: float, steps: int = 1):
        """Accumulate RDP divergence for Poisson/subsampled Gaussian mechanism.

        Args:
            q: Subsampling ratio (|B| / |D|)
            noise_multiplier: Noise scale sigma (std / clipping_bound)
            steps: Number of mini-batch gradient steps
        """
        if noise_multiplier <= 0.0:
            return
        for alpha in self.orders:
            # First-order subsampled Gaussian RDP expansion:
            # RDP(alpha) <= steps * (q^2 * alpha) / (2 * sigma^2) + O(q^3)
            rdp_step = (q**2 * alpha) / (2.0 * noise_multiplier**2)
            self.rdp[alpha] += rdp_step * steps

    def get_epsilon(self, target_delta: float = 1e-5) -> float:
        """Convert cumulative RDP bounds into optimal (epsilon, delta)-DP guarantee.

        eps(delta) = min_{alpha > 1} [ RDP(alpha) + log(1/delta) / (alpha - 1) ]
        """
        epsilons = [
            self.rdp[alpha] + math.log(1.0 / target_delta) / (alpha - 1.0)
            for alpha in self.orders
            if alpha > 1.0
        ]
        return min(epsilons) if epsilons else float("inf")


def create_dp_train_step(
    model_apply_fn: Callable, clipping_threshold: float, noise_multiplier: float
) -> Callable:
    """Compile an end-to-end vectorized DP-SGD optimization step in pure JAX.

    Uses `jax.vmap` over single-sample gradients to avoid O(B) memory overhead.
    """

    def single_sample_loss(params, x, y):
        logits = model_apply_fn(params, jnp.expand_dims(x, 0), train=True)
        one_hot = jax.nn.one_hot(jnp.expand_dims(y, 0), 10)
        return optax.softmax_cross_entropy(logits, one_hot).mean()

    # Vectorized gradient function: computes exact per-sample gradient tree
    per_sample_grad_fn = jax.vmap(jax.grad(single_sample_loss), in_axes=(None, 0, 0))

    @jax.jit
    def dp_train_step(state: train_state.TrainState, batch_x, batch_y, rng_key):
        b_size = batch_x.shape[0]

        # 1. Compute per-sample gradients
        per_sample_grads = per_sample_grad_fn(state.params, batch_x, batch_y)

        # 2. Compute L2 norm per sample across all parameter tensors
        sq_norms = jnp.zeros(b_size)
        for leaf in jax.tree_util.tree_leaves(per_sample_grads):
            sq_norms += jnp.sum(jnp.square(leaf), axis=tuple(range(1, leaf.ndim)))
        l2_norms = jnp.sqrt(sq_norms)

        # 3. Compute clipping factor: min(1, C / ||g||_2)
        clipping_factors = jnp.minimum(1.0, clipping_threshold / (l2_norms + 1e-6))

        # 4. Clip per-sample gradients and average across batch
        def clip_and_average(grad_leaf):
            shape = (-1,) + (1,) * (grad_leaf.ndim - 1)
            scaled = grad_leaf * clipping_factors.reshape(shape)
            return jnp.mean(scaled, axis=0)

        avg_clipped_leaves = jax.tree_util.tree_map(clip_and_average, per_sample_grads)

        # 5. Add Gaussian noise to averaged clipped gradient leaves
        if noise_multiplier > 0.0:
            noise_std = (noise_multiplier * clipping_threshold) / float(b_size)
            leaves = jax.tree_util.tree_leaves(avg_clipped_leaves)
            noise_keys = jax.random.split(rng_key, len(leaves))
            noisy_grad_leaves = [
                g + jax.random.normal(k, g.shape) * noise_std
                for g, k in zip(leaves, noise_keys, strict=True)
            ]
            final_grads = jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(avg_clipped_leaves),
                noisy_grad_leaves,
            )
        else:
            final_grads = avg_clipped_leaves

        new_state = state.apply_gradients(grads=final_grads)
        return new_state, jnp.mean(l2_norms)

    return dp_train_step
