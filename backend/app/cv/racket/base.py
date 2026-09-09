"""Racket detector protocol and raw detection types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.schemas.pose import PoseSequence


@dataclass(frozen=True, slots=True)
class RawRacketDetection:
    """Detector-native racket observation before conversion to RacketPoint.

    Pixel-space box and representative point (typically tip / head center).
    """

    frame_index: int
    x_px: float | None
    y_px: float | None
    x1_px: float | None
    y1_px: float | None
    x2_px: float | None
    y2_px: float | None
    confidence: float
    hand: str | None = None


class RacketDetector(ABC):
    """Independent racket detector — must not call MMPose or shuttle trackers."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when runtime deps / weights are ready."""

    @abstractmethod
    def detect(
        self,
        video_path: Path,
        *,
        pose: PoseSequence | None = None,
        hitting_hand: str | None = None,
    ) -> list[RawRacketDetection]:
        """Detect the primary player's racket per frame."""
