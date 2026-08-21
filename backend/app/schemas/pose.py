"""Internal pose schema — MMPose-agnostic types used across the app."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Keypoint:
    """Normalized joint in image space: x/y in [0, 1], plus confidence."""

    x: float
    y: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "confidence": self.confidence}


@dataclass(slots=True)
class PoseFrame:
    frame_index: int
    timestamp: float
    keypoints: dict[str, Keypoint] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "keypoints": {name: kp.to_dict() for name, kp in self.keypoints.items()},
        }


@dataclass(slots=True)
class PoseSequence:
    video: str
    frames: list[PoseFrame] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def append(self, frame: PoseFrame) -> None:
        self.frames.append(frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "frame_count": self.frame_count,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
