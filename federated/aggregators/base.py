"""Abstract base contract for federated learning aggregation algorithms."""

from abc import ABC, abstractmethod
from typing import Any


class BaseAggregator(ABC):
    """Abstract base class for federated aggregation algorithms from research literature."""

    def __init__(self, name: str) -> None:
        self.name = name

    def reset(self) -> None:  # noqa: B027
        """Reset internal round-to-round states (e.g. server momentum buffers)."""
        pass

    @property
    def requires_proximal_loss(self) -> bool:
        """Whether client local training should include a proximal regularization term."""
        return False

    @property
    def proximal_mu(self) -> float:
        """Proximal regularization coefficient mu for local training."""
        return 0.0

    @abstractmethod
    def aggregate(
        self,
        server_params: Any,
        client_params: list[Any],
        client_weights: list[float] | None = None,
    ) -> Any:
        """Aggregate client parameter updates into new global server parameters.

        Parameters
        ----------
        server_params : Any
            Current global parameters (JAX PyTree) at the beginning of the round.
        client_params : list[Any]
            List of updated parameter PyTrees from participating clients.
        client_weights : Optional[list[float]]
            Relative sample counts or weighting coefficients for each client.

        Returns
        -------
        Any
            Aggregated global parameters PyTree for the next round.
        """
        pass
