"""Dynamic model registration, discovery, and instantiation factory."""

import flax.linen as nn

from models.alexnet import AlexNet
from models.densenet import DenseNet169
from models.googlenet import GoogLeNet
from models.inception import InceptionV3
from models.mobilenet import MobileNetV2
from models.resnet import ResNet9, ResNet18, ResNet50
from models.shufflenet import ShuffleNetV2
from models.simple_cnn import SimpleCNN
from models.titans import VisionTitans
from models.ttt import VisionTTT
from models.vgg import VGG16, VGG19
from models.vit import VisionTransformer, ViTSmall, ViTTiny

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "simple_cnn": SimpleCNN,
    "simplecnn": SimpleCNN,
    "alexnet": AlexNet,
    "vgg16": VGG16,
    "vgg": VGG16,
    "vgg19": VGG19,
    "vgg_19": VGG19,
    "resnet18": ResNet18,
    "resnet": ResNet18,
    "resnet50": ResNet50,
    "resnet_50": ResNet50,
    "resnet9": ResNet9,
    "resnet_9": ResNet9,
    "fast_resnet": ResNet9,
    "mobilenetv2": MobileNetV2,
    "mobilenet": MobileNetV2,
    "shufflenetv2": ShuffleNetV2,
    "shufflenet_v2": ShuffleNetV2,
    "shufflenet": ShuffleNetV2,
    "googlenet": GoogLeNet,
    "inception_v1": GoogLeNet,
    "inceptionv1": GoogLeNet,
    "inceptionv3": InceptionV3,
    "inception_v3": InceptionV3,
    "densenet169": DenseNet169,
    "densenet_169": DenseNet169,
    "densenet": DenseNet169,
    "vit": VisionTransformer,
    "vision_transformer": VisionTransformer,
    "vit_tiny": ViTTiny,
    "vit_small": ViTSmall,
    "titans": VisionTitans,
    "vision_titans": VisionTitans,
    "ttt": VisionTTT,
    "vision_ttt": VisionTTT,
    "ttt_linear": VisionTTT,
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
    import inspect

    key = name.strip().lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list_models()}")
    cls = MODEL_REGISTRY[key]

    # Filter kwargs to only those accepted by model constructor
    sig = inspect.signature(cls)
    valid_params = sig.parameters
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in valid_params.values()
    )
    if has_var_kw:
        filtered_kwargs = dict(kwargs)
    else:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    if "num_classes" in valid_params:
        filtered_kwargs["num_classes"] = num_classes

    return cls(**filtered_kwargs)


def list_models() -> list[str]:
    """Return a list of canonical model names available in the registry."""
    return [
        "simple_cnn",
        "alexnet",
        "resnet9",
        "resnet18",
        "resnet50",
        "mobilenetv2",
        "shufflenetv2",
        "googlenet",
        "inceptionv3",
        "densenet169",
        "vgg16",
        "vgg19",
        "vit",
        "titans",
        "ttt",
    ]
