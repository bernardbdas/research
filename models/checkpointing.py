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

    Supports both keyword-based directory layout and direct file path invocation:
    - save_model_checkpoint(params, checkpoint_dir=..., model_name=..., dataset_name=..., strategy=..., history=...)
    - save_model_checkpoint(target_file_path, history, params)
    """
    if isinstance(params, str | Path) and (
        str(params).endswith(".msgpack") or "/" in str(params)
    ):
        file_path = Path(params)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Invocations passing save_model_checkpoint(path, history, params)
        actual_hist = (
            checkpoint_dir if isinstance(checkpoint_dir, dict) else (history or {})
        )
        actual_params = model_name if not isinstance(model_name, str) else params
        # If 3rd arg was params
        if isinstance(dataset_name, dict) and actual_params is None:
            actual_params = dataset_name
        payload = {
            "params": actual_params,
            "model_name": file_path.stem.split("_")[0]
            if "_" in file_path.stem
            else "model",
            "dataset_name": file_path.stem.split("_")[1]
            if len(file_path.stem.split("_")) > 1
            else "data",
            "strategy": file_path.stem.split("_")[-1]
            if "_" in file_path.stem
            else "strategy",
            "history": actual_hist,
            "extra_metadata": extra_metadata or {},
            "saved_at": time.time(),
        }
        serialized_bytes = flax.serialization.msgpack_serialize(payload)
        with open(file_path, "wb") as f:
            f.write(serialized_bytes)
        return file_path

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
