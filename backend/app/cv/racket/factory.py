"""Factory for racket detector backends."""

from __future__ import annotations

from app.config import settings
from app.cv.racket.base import RacketDetector
from app.cv.racket.pose_guided import PoseGuidedRacketDetector
from app.cv.racket.yolo import YoloRacketDetector


def get_racket_detector(backend: str | None = None) -> RacketDetector:
    name = (backend or settings.racket_backend or "pose_guided").strip().lower()
    if name in ("pose_guided", "pose", "heuristic"):
        return PoseGuidedRacketDetector()
    if name in ("yolo", "ultralytics"):
        return YoloRacketDetector()
    raise ValueError(f"Unknown racket backend: {name}")
