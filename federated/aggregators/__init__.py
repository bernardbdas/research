"""Pluggable federated aggregation algorithms from research literature."""

from federated.aggregators.base import BaseAggregator
from federated.aggregators.fedadam import FedAdam
from federated.aggregators.fedavg import FedAvg
from federated.aggregators.fedm3 import FedM3
from federated.aggregators.fedmedian import FedMedian
from federated.aggregators.fedmuon import FedMuon
from federated.aggregators.fedprox import FedProx
from federated.aggregators.fedtrimmedmean import FedTrimmedMean
from federated.aggregators.registry import (
    AGGREGATOR_REGISTRY,
    get_aggregator,
    list_aggregators,
    register_aggregator,
)

__all__ = [
    "AGGREGATOR_REGISTRY",
    "BaseAggregator",
    "FedAdam",
    "FedAvg",
    "FedM3",
    "FedMedian",
    "FedMuon",
    "FedProx",
    "FedTrimmedMean",
    "get_aggregator",
    "list_aggregators",
    "register_aggregator",
]
