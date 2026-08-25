"""DensePose integration for body-surface muscle overlay."""

# Must run before Detectron2 / densepose imports (Pillow 10+ removed Image.LINEAR).
from app.cv.densepose.compat import apply_pillow_shims

apply_pillow_shims()

from app.cv.densepose.debug import DensePoseDebugSession
from app.cv.densepose.inferencer import DensePoseError, DensePoseInferencer
from app.cv.densepose.mapping import DensePoseFrameResult, DensePosePart

__all__ = [
    "DensePoseDebugSession",
    "DensePoseError",
    "DensePoseFrameResult",
    "DensePoseInferencer",
    "DensePosePart",
]
