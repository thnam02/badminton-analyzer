"""Shim so mmpose can import xtcocotools without building the C extension."""

from pycocotools.coco import COCO

__all__ = ["COCO"]
