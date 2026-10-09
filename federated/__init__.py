"""Federated Learning module with plug-and-play aggregation strategies and training simulation."""

from federated import aggregators
from federated.aggregators import (
    AGGREGATOR_REGISTRY,
    BaseAggregator,
    FedAdam,
    FedAvg,
    FedMedian,
    FedMuon,
    FedProx,
    FedTrimmedMean,
    get_aggregator,
    list_aggregators,
    register_aggregator,
)
from federated.training import (
    extract_partition_arrays,
    make_eval_step,
    make_train_step,
    train_federated,
)

# Friendly alias for researchers preferring 'strategies' nomenclature
strategies = aggregators

__all__ = [
    "AGGREGATOR_REGISTRY",
    "BaseAggregator",
    "FedAdam",
    "FedAvg",
    "FedMedian",
    "FedMuon",
    "FedProx",
    "FedTrimmedMean",
    "aggregators",
    "extract_partition_arrays",
    "get_aggregator",
    "list_aggregators",
    "make_eval_step",
    "make_train_step",
    "register_aggregator",
    "strategies",
    "train_federated",
]
