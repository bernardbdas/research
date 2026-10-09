"""Federated learning training and simulation sub-package."""

from federated.training.dataset import (
    _extract_partition_arrays,
    extract_partition_arrays,
)
from federated.training.simulation import train_federated
from federated.training.steps import (
    _make_eval_step,
    _make_train_step,
    make_eval_step,
    make_train_step,
)

__all__ = [
    "_extract_partition_arrays",
    "_make_eval_step",
    "_make_train_step",
    "extract_partition_arrays",
    "make_eval_step",
    "make_train_step",
    "train_federated",
]
