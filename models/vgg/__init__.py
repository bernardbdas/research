"""VGG architecture sub-package."""

from models.vgg.backbone import VGG, VGG16, VGG19
from models.vgg.configs import CFG_VGG

__all__ = ["CFG_VGG", "VGG", "VGG16", "VGG19"]
