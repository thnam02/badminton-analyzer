"""Internal shuttlecock trajectory schemas (tracker-agnostic)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ShuttlePoint:
    """One frame of shuttle location in normalized image coordinates."""

    frame_index: int
    timestamp: float
    x: float | None = None
    y: float | None = None
    visible: bool = False
    confidence: float = 0.0
    interpolated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "x": self.x,
            "y": self.y,
            "visible": self.visible,
            "confidence": self.confidence,
            "interpolated": self.interpolated,
        }


@dataclass(slots=True)
class ShuttleTrajectory:
    video: str
    backend: str
    fps: float
    width: int
    height: int
    frames: list[ShuttlePoint] = field(default_factory=list)
    missing_frame_indices: list[int] = field(default_factory=list)
    notes: str = (
        "Shuttle trajectory from an independent tracker adapter. "
        "Does not alter pose or estimated-contact logic."
    )

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def visible_count(self) -> int:
        return sum(1 for f in self.frames if f.visible and not f.interpolated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "backend": self.backend,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "visible_count": self.visible_count,
            "missing_frame_indices": list(self.missing_frame_indices),
            "notes": self.notes,
            "frames": [f.to_dict() for f in self.frames],
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
