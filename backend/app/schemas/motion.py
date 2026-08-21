"""Motion-derivative schemas (speeds / angular velocities + peaks)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PeakStats:
    value: float | None
    frame_index: int | None
    timestamp: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class MotionFrame:
    frame_index: int
    timestamp: float
    # Normalized image units per second (Euclidean).
    right_wrist_speed: float | None = None
    # Degrees per second (signed); None when either sample or dt is invalid.
    right_elbow_angular_velocity: float | None = None
    right_knee_angular_velocity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "right_wrist_speed": self.right_wrist_speed,
            "right_elbow_angular_velocity": self.right_elbow_angular_velocity,
            "right_knee_angular_velocity": self.right_knee_angular_velocity,
        }


@dataclass(slots=True)
class MotionSequence:
    video: str
    frames: list[MotionFrame] = field(default_factory=list)
    peaks: dict[str, PeakStats] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def append(self, frame: MotionFrame) -> None:
        self.frames.append(frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "frame_count": self.frame_count,
            "frames": [frame.to_dict() for frame in self.frames],
            "peaks": {name: peak.to_dict() for name, peak in self.peaks.items()},
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
