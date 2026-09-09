"""Internal racket trajectory schemas (detector-agnostic)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RacketBBox:
    """Axis-aligned box in normalized image coordinates [0, 1]."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)


@dataclass(slots=True)
class RacketPoint:
    """One frame of racket state for the primary player."""

    frame_index: int
    timestamp: float
    x: float | None = None
    y: float | None = None
    bbox: RacketBBox | None = None
    visible: bool = False
    confidence: float = 0.0
    hand: str | None = None  # "LEFT" | "RIGHT"
    tracked: bool = False
    interpolated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "x": self.x,
            "y": self.y,
            "bbox": self.bbox.to_dict() if self.bbox is not None else None,
            "visible": self.visible,
            "confidence": self.confidence,
            "hand": self.hand,
            "tracked": self.tracked,
            "interpolated": self.interpolated,
        }


@dataclass(slots=True)
class RacketTrajectory:
    video: str
    backend: str
    fps: float
    width: int
    height: int
    hitting_hand: str | None = None
    frames: list[RacketPoint] = field(default_factory=list)
    missing_frame_indices: list[int] = field(default_factory=list)
    notes: str = (
        "Racket trajectory from an independent detector/tracker. "
        "Optional pose context associates the racket with the hitting hand. "
        "Does not alter pose, shuttle, or estimated-contact logic."
    )

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def visible_count(self) -> int:
        return sum(
            1
            for f in self.frames
            if f.visible and not f.interpolated
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "backend": self.backend,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "hitting_hand": self.hitting_hand,
            "frame_count": self.frame_count,
            "visible_count": self.visible_count,
            "missing_frame_indices": list(self.missing_frame_indices),
            "notes": self.notes,
            "frames": [f.to_dict() for f in self.frames],
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
