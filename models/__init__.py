"""Model backbones for Federated Learning experiments in Flax/JAX."""

from typing import Any

import flax.linen as nn
import jax

from models.mobilenet import MobileNetV2
from models.resnet import ResNet, ResNet18
from models.simple_cnn import SimpleCNN
from models.vgg import VGG, VGG16

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "simple_cnn": SimpleCNN,
    "simplecnn": SimpleCNN,
    "vgg16": VGG16,
    "vgg": VGG16,
    "resnet18": ResNet18,
    "resnet": ResNet18,
    "mobilenetv2": MobileNetV2,
    "mobilenet": MobileNetV2,
}


def get_model(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """Factory function to dynamically instantiate model backbones by name.

    Parameters
    ----------
    name : str
        Name of the model (e.g. 'simple_cnn', 'resnet18', 'mobilenetv2', 'vgg16').
    num_classes : int
        Number of output classes. Default is 10.
    **kwargs : Any
        Additional keyword arguments passed to the model constructor.

    Returns
    -------
    nn.Module
        Instantiated Flax neural network module.
    """
    key = name.strip().lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list_models()}")
    cls = MODEL_REGISTRY[key]
    return cls(num_classes=num_classes, **kwargs)


def list_models() -> list[str]:
    """Return a list of canonical model names available in the registry."""
    return ["simple_cnn", "resnet18", "mobilenetv2", "vgg16"]


def count_parameters(variables: dict[str, Any]) -> int:
    """Calculate the total number of trainable parameters in a Flax parameter PyTree."""
    params = variables.get("params", variables)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


__all__ = [
    "SimpleCNN",
    "VGG16",
    "VGG",
    "ResNet18",
    "ResNet",
    "MobileNetV2",
    "MODEL_REGISTRY",
    "get_model",
    "list_models",
    "count_parameters",
]
