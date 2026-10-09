"""MobileNetV2 architecture sub-package."""

from models.mobilenet.backbone import MobileNetV2
from models.mobilenet.blocks import InvertedResidual

__all__ = ["InvertedResidual", "MobileNetV2"]
