"""Class-incremental learning components: feature extractors and dynamic classifier heads."""

from models.continual.classifier import (
    IncrementalClassifier,
    expand_head_parameters,
)
from models.continual.extractor import FlaxFeatureExtractor

__all__ = [
    "FlaxFeatureExtractor",
    "IncrementalClassifier",
    "expand_head_parameters",
]
