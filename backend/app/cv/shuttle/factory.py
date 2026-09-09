"""Factory for shuttle tracker backends."""

from __future__ import annotations

from app.config import settings
from app.cv.shuttle.base import ShuttleTracker
from app.cv.shuttle.heuristic import HeuristicShuttleTracker
from app.cv.shuttle.tracknet_v3 import TrackNetV3Adapter


def get_shuttle_tracker(backend: str | None = None) -> ShuttleTracker:
    name = (backend or settings.shuttle_backend or "tracknetv3").strip().lower()
    if name in ("tracknetv3", "tracknet", "tnv3"):
        return TrackNetV3Adapter()
    if name in ("heuristic", "diff", "blob"):
        return HeuristicShuttleTracker()
    raise ValueError(f"Unknown shuttle backend: {name}")
