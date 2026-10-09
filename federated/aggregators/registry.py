"""Pluggable registry and factory for federated aggregation algorithms."""

from collections.abc import Callable
from typing import Any

from federated.aggregators.base import BaseAggregator

AGGREGATOR_REGISTRY: dict[str, type[BaseAggregator]] = {}


def register_aggregator(
    *names: str,
) -> Callable[[type[BaseAggregator]], type[BaseAggregator]]:
    """Decorator to register a new research paper aggregation algorithm into the registry.

    Parameters
    ----------
    *names : str
        One or more canonical names or aliases for the algorithm (e.g. 'fedavg', 'fed_avg').

    Returns
    -------
    Callable[[type[BaseAggregator]], type[BaseAggregator]]
        Decorated class.
    """

    def decorator(cls: type[BaseAggregator]) -> type[BaseAggregator]:
        for name in names:
            key = name.strip().lower().replace("-", "_")
            AGGREGATOR_REGISTRY[key] = cls
        return cls

    return decorator


def get_aggregator(
    aggregator: BaseAggregator | str | None = None,
    **kwargs: Any,
) -> BaseAggregator:
    """Resolve and instantiate a federated aggregator strategy.

    Parameters
    ----------
    aggregator : BaseAggregator | str | None
        An aggregator instance or strategy name (e.g. 'fedavg', 'fedprox', 'fedmedian', 'fedadam').
        Defaults to FedAvg() if None.
    **kwargs : Any
        Hyperparameters passed to the aggregator constructor (e.g. mu=0.01 for FedProx,
        beta=0.1 for FedTrimmedMean, eta=0.05 for FedAdam).

    Returns
    -------
    BaseAggregator
        Configured aggregator instance.
    """
    if aggregator is None:
        aggregator_cls = AGGREGATOR_REGISTRY.get("fedavg")
        if aggregator_cls is None:
            from federated.aggregators.fedavg import FedAvg

            return FedAvg(**kwargs)
        return aggregator_cls(**kwargs)

    if isinstance(aggregator, BaseAggregator):
        return aggregator

    key = str(aggregator).strip().lower().replace("-", "_")
    if key not in AGGREGATOR_REGISTRY:
        options = ", ".join(sorted(set(AGGREGATOR_REGISTRY.keys())))
        raise ValueError(
            f"Unknown aggregator '{aggregator}'. Supported options: {options}"
        )

    cls = AGGREGATOR_REGISTRY[key]
    return cls(**kwargs)


def list_aggregators() -> list[str]:
    """Return canonical names of available federated aggregation algorithms."""
    canonical = [
        "fedavg",
        "fedprox",
        "fedmedian",
        "fedtrimmedmean",
        "fedadam",
        "fedmuon",
    ]
    seen_classes = set()
    result = []
    for name in canonical:
        cls = AGGREGATOR_REGISTRY.get(name)
        if cls:
            seen_classes.add(cls)
            result.append(name)
    for name, cls in sorted(AGGREGATOR_REGISTRY.items()):
        if cls not in seen_classes:
            seen_classes.add(cls)
            result.append(name)
    return result
