"""GoogLeNet architecture sub-package."""

from models.googlenet.backbone import GoogLeNet
from models.googlenet.blocks import InceptionBlock

__all__ = ["GoogLeNet", "InceptionBlock"]
