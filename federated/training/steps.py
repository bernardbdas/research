"""JIT-compiled gradient and evaluation step generators for federated rounds."""

from typing import Any

import jax
import jax.numpy as jnp
import optax


def make_train_step(apply_fn: Any, mu: float = 0.0) -> Any:
    """Create a JIT-compiled gradient step with optional FedProx proximal regularization.

    Parameters
    ----------
    apply_fn : Any
        Model forward apply function.
    mu : float
        Proximal regularization coefficient mu. Default is 0.0.

    Returns
    -------
    Callable
        JIT-compiled training step function.
    """

    @jax.jit
    def train_step(state: Any, batch_x: Any, batch_y: Any, global_params: Any = None):
        def loss_fn(params: Any):
            logits = apply_fn({"params": params}, batch_x, train=True)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch_y
            )
            base_loss = jnp.mean(loss)
            if mu > 0.0 and global_params is not None:
                prox_term = sum(
                    jnp.sum((p - gp) ** 2)
                    for p, gp in zip(
                        jax.tree_util.tree_leaves(params),
                        jax.tree_util.tree_leaves(global_params),
                        strict=True,
                    )
                )
                return base_loss + 0.5 * mu * prox_term
            return base_loss

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads)

    return train_step


def make_eval_step(apply_fn: Any) -> Any:
    """Create a JIT-compiled evaluation step for accuracy and loss.

    Parameters
    ----------
    apply_fn : Any
        Model forward apply function.

    Returns
    -------
    Callable
        JIT-compiled evaluation step function.
    """

    @jax.jit
    def eval_step(params: Any, batch_x: Any, batch_y: Any) -> tuple[Any, Any]:
        logits = apply_fn({"params": params}, batch_x, train=False)
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch_y
            )
        )
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == batch_y)
        return loss, acc

    return eval_step


# Backward compatibility aliases
_make_train_step = make_train_step
_make_eval_step = make_eval_step
