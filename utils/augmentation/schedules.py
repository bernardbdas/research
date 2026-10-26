"""Learning rate scheduling utilities."""

import optax


def create_cosine_schedule(
    total_epochs: int,
    steps_per_epoch: int,
    base_lr: float = 0.05,
    min_lr: float = 1e-4,
    warmup_epochs: int = 2,
) -> optax.Schedule:
    """Create a Cosine Annealing learning rate schedule with linear warmup.

    Parameters
    ----------
    total_epochs : int
        Total training epochs.
    steps_per_epoch : int
        Number of optimization steps per epoch (n_samples // batch_size).
    base_lr : float
        Peak learning rate after warmup. Default is 0.05.
    min_lr : float
        Minimum learning rate at the end of training. Default is 1e-4.
    warmup_epochs : int
        Number of epochs spent linearly warming up from min_lr to base_lr. Default is 2.

    Returns
    -------
    optax.Schedule
        Configured Optax learning rate schedule callable.
    """
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = total_epochs * steps_per_epoch
    decay_steps = max(warmup_steps + 1, total_steps)

    return optax.warmup_cosine_decay_schedule(
        init_value=min_lr,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=min_lr,
    )
