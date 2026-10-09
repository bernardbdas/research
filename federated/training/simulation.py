"""Multi-round federated training simulation engine."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flwr_datasets import FederatedDataset

from federated.aggregators import BaseAggregator, get_aggregator
from federated.training.dataset import extract_partition_arrays
from federated.training.steps import make_eval_step, make_train_step
from utils.data_processing import load_processed_dataset


def train_federated(
    fds: FederatedDataset,
    dataset_name: str,
    aggregator: BaseAggregator | str = "fedavg",
    num_clients: int | None = None,
    rounds: int = 10,
    num_rounds: int | None = None,
    local_epochs: int = 1,
    batch_size: int = 64,
    lr: float = 1e-3,
    model: Any | None = None,
    max_client_samples: int = 1000,
    max_test_samples: int = 1000,
    seed: int = 42,
    verbose: bool = True,
    **aggregator_kwargs: Any,
) -> tuple[dict[str, list[Any]], Any]:
    """Execute multi-round federated training with plug-and-play aggregation strategies.

    Parameters
    ----------
    fds : FederatedDataset
        Flower FederatedDataset partitioned across simulated clients.
    dataset_name : str
        Name or alias of the benchmark dataset (e.g. 'mnist', 'cifar10').
    aggregator : BaseAggregator | str
        Aggregation strategy instance or algorithm name:
        'fedavg', 'fedprox', 'fedmedian', 'fedtrimmedmean', 'fedadam'.
        Defaults to 'fedavg'.
    num_clients : Optional[int]
        Number of participating federated clients. If None, auto-detected from fds partitioner.
    rounds : int
        Number of communication rounds. Default is 10.
    num_rounds : Optional[int]
        Alias for rounds (supports backward compatibility).
    local_epochs : int
        Number of local training epochs on each client per round. Default is 1.
    batch_size : int
        Local training mini-batch size. Default is 64.
    lr : float
        Client-side learning rate for local AdamW optimizer. Default is 1e-3.
    model : Optional[Any]
        Flax model instance. Defaults to SimpleCNN(num_classes=10).
    max_client_samples : int
        Upper bound on samples extracted per client partition. Default is 1000.
    max_test_samples : int
        Upper bound on samples used for global evaluation. Default is 1000.
    seed : int
        Random seed for model initialization and client shuffling. Default is 42.
    verbose : bool
        If True, prints per-round test accuracy and loss. Default is True.
    **aggregator_kwargs : Any
        Hyperparameters forwarded to the aggregator if specified by name
        (e.g., mu=0.01 for fedprox, beta=0.1 for fedtrimmedmean, eta=0.05 for fedadam).

    Returns
    -------
    tuple[dict[str, list], Any]
        History dictionary with keys 'round', 'acc', 'loss', and the final trained parameters.
    """
    strat = get_aggregator(aggregator, **aggregator_kwargs)
    strat.reset()

    if num_rounds is not None:
        rounds = num_rounds

    if num_clients is None:
        if hasattr(fds, "partitioners") and "train" in fds.partitioners:
            part = fds.partitioners["train"]
            num_clients = getattr(part, "num_partitions", 5)
        else:
            num_clients = 5

    is_cifar = "cifar" in dataset_name.lower()
    if model is None:
        if is_cifar:
            from models.resnet import ResNet50

            model = ResNet50(num_classes=10)
        else:
            from models.vgg import VGG16

            model = VGG16(num_classes=10)

    # 1. Extract client data partitions
    img_key = "img" if is_cifar else "image"
    client_data: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    client_weights: list[float] = []

    for i in range(num_clients):
        part = fds.load_partition(i, split="train")
        cx, cy = extract_partition_arrays(part, img_key, max_client_samples)
        if is_cifar:
            from utils.augmentation import normalize_cifar10

            cx = jnp.array(normalize_cifar10(np.array(cx)))
        client_data.append((cx, cy))
        client_weights.append(float(len(cx)))

    # 2. Load centralized evaluation test split
    test_ds = load_processed_dataset(dataset_name, split="test")
    test_x, test_y = extract_partition_arrays(test_ds, img_key, max_test_samples)
    if is_cifar:
        from utils.augmentation import normalize_cifar10

        test_x = jnp.array(normalize_cifar10(np.array(test_x)))

    # 3. Initialize global model parameters
    key = jax.random.PRNGKey(seed)
    params = model.init(key, client_data[0][0][:2], train=False)["params"]

    # 4. Compile train and evaluation step functions
    prox_mu = strat.proximal_mu if strat.requires_proximal_loss else 0.0
    step_fn = make_train_step(model.apply, mu=prox_mu)
    eval_fn = make_eval_step(model.apply)

    history: dict[str, list[Any]] = {"round": [], "acc": [], "loss": []}

    # 5. Multi-round federated training loop
    for rnd in range(1, rounds + 1):
        local_params: list[Any] = []

        for cx, cy in client_data:
            state = train_state.TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=optax.adamw(lr),
            )
            n = len(cx)
            for _ in range(local_epochs):
                perm = np.random.permutation(n)
                for s in range(0, n, batch_size):
                    idx = perm[s : s + batch_size]
                    state = step_fn(state, cx[idx], cy[idx], global_params=params)
            local_params.append(state.params)

        # 6. Plug-and-play parameter aggregation
        params = strat.aggregate(
            server_params=params,
            client_params=local_params,
            client_weights=client_weights,
        )

        # 7. Global test evaluation
        loss, acc = eval_fn(params, test_x, test_y)
        acc_pct = float(acc) * 100.0
        loss_val = float(loss)

        history["round"].append(rnd)
        history["acc"].append(acc_pct)
        history["loss"].append(loss_val)

        if verbose:
            print(
                f"  [{strat.name}] Round {rnd:02d} -> "
                f"Test Acc: {acc_pct:5.2f}% | Loss: {loss_val:.4f}"
            )

    return history, params
