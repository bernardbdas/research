"""Model backbones and architectures for Federated Learning in Flax/JAX."""

from models.alexnet import AlexNet
from models.common import count_parameters
from models.densenet import DenseNet, DenseNet169
from models.googlenet import GoogLeNet
from models.inception import InceptionV3
from models.mobilenet import MobileNetV2
from models.registry import MODEL_REGISTRY, get_model, list_models
from models.resnet import ResNet, ResNet9, ResNet18, ResNet50
from models.shufflenet import ShuffleNetV2
from models.simple_cnn import SimpleCNN
from models.titans import VisionTitans
from models.ttt import VisionTTT
from models.vgg import VGG, VGG16, VGG19
from models.vit import VisionTransformer, ViTSmall, ViTTiny

__all__ = [
    "AlexNet",
    "DenseNet",
    "DenseNet169",
    "GoogLeNet",
    "InceptionV3",
    "MODEL_REGISTRY",
    "MobileNetV2",
    "ResNet",
    "ResNet9",
    "ResNet18",
    "ResNet50",
    "ShuffleNetV2",
    "SimpleCNN",
    "VGG",
    "VGG16",
    "VGG19",
    "VisionTTT",
    "VisionTitans",
    "VisionTransformer",
    "ViTSmall",
    "ViTTiny",
    "count_parameters",
    "get_model",
    "list_models",
]
