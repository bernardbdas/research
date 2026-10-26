"""HOPE-FL module: Continuous Memory System (CMS) and Federated Predictive Architectures."""

from models.hope.backbone import HOPEFLNet
from models.hope.cms import ContinuousMemorySystem

__all__ = ["ContinuousMemorySystem", "HOPEFLNet"]
