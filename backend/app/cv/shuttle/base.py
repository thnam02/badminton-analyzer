"""Shuttle tracker protocol and raw detection types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RawShuttleDetection:
    """Tracker-native detection before conversion to ShuttlePoint.

    Coordinates are in pixel space of the source video.
    ``visibility`` follows TrackNet convention: 0 = missing, >0 = present.
    """

    frame_index: int
    x_px: float | None
    y_px: float | None
    visibility: int
    confidence: float = 0.0


class ShuttleTracker(ABC):
    """Independent shuttlecock tracker — must not touch the pose pipeline."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when weights / runtime deps are ready."""

    @abstractmethod
    def track(self, video_path: Path) -> list[RawShuttleDetection]:
        """Run tracking on a video; return per-frame raw detections."""
