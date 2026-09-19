"""Joint-angle schemas derived from PoseSequence (degrees, nullable)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AngleFrame:
    frame_index: int
    timestamp: float
    # Degrees when computable; None when required keypoints fail confidence / missing.
    right_elbow: float | None = None
    right_knee: float | None = None
    right_shoulder: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "angles": {
                "right_elbow": self.right_elbow,
                "right_knee": self.right_knee,
                "right_shoulder": self.right_shoulder,
            },
        }


@dataclass(slots=True)
class AngleSequence:
    video: str
    frames: list[AngleFrame] = field(default_factory=list)
    analysis_id: str = ""
    snapshot_id: str = ""
    fingerprint: str = ""
    snapshot_schema_version: str = ""
    artifact_schema_version: str = ""
    artifact_role: str = ""

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def append(self, frame: AngleFrame) -> None:
        self.frames.append(frame)

    def to_dict(self) -> dict[str, Any]:
        from app.schemas.provenance import provenance_fields_from_object

        payload: dict[str, Any] = {
            "video": self.video,
            "frame_count": self.frame_count,
            "frames": [frame.to_dict() for frame in self.frames],
        }
        payload.update(provenance_fields_from_object(self))
        return payload

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
