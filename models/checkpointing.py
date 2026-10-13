"""Model checkpointing and serialization utilities for Flax PyTrees."""

import time
from pathlib import Path
from typing import Any

import flax.serialization


def save_model_checkpoint(
    params: Any,
    checkpoint_dir: str | Path = "checkpoints",
    model_name: str = "simple_cnn",
    dataset_name: str = "mnist",
    strategy: str = "centralized",
    history: dict[str, list[Any]] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Serialize and save Flax model parameters, architecture metadata, and training history.

    Parameters
    ----------
    params : Any
        Flax model parameter PyTree.
    checkpoint_dir : str | Path
        Target directory to store checkpoint files. Default is 'checkpoints'.
    model_name : str
        Canonical name of the model backbone architecture (e.g. 'simple_cnn', 'resnet18').
    dataset_name : str
        Name of the dataset trained on (e.g. 'mnist', 'cifar10').
    strategy : str
        Training strategy identifier (e.g. 'centralized', 'fedavg', 'fedprox').
    history : Optional[dict[str, list]]
        Training metric trajectory history (epochs/rounds, accuracy, loss).
    extra_metadata : Optional[dict[str, Any]]
        Arbitrary extra metadata (learning rate, optimizer, seed, etc.).

    Returns
    -------
    Path
        Absolute path to the saved checkpoint file.
    """
    dest_dir = Path(checkpoint_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name}_{dataset_name}_{strategy}.msgpack"
    file_path = dest_dir / filename

    payload = {
        "params": params,
        "model_name": str(model_name),
        "dataset_name": str(dataset_name),
        "strategy": str(strategy),
        "history": history or {},
        "extra_metadata": extra_metadata or {},
        "saved_at": time.time(),
    }

    serialized_bytes = flax.serialization.msgpack_serialize(payload)
    with open(file_path, "wb") as f:
        f.write(serialized_bytes)

    return file_path


def load_model_checkpoint(path: str | Path) -> dict[str, Any]:
    """Deserialize a saved Flax model checkpoint.

    Parameters
    ----------
    path : str | Path
        Path to the saved `.msgpack` checkpoint file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'params', 'model_name', 'dataset_name', 'strategy',
        'history', and 'extra_metadata'.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {file_path}")

    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    restored = flax.serialization.msgpack_restore(raw_bytes)
    return restored
