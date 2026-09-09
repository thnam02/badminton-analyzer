"""Representative smash keyframe schemas (no CV inference / OpenAI)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Contact neighbor labels (not SmashPhase members).
CONTACT_MINUS_2 = "CONTACT_MINUS_2"
CONTACT_PLUS_2 = "CONTACT_PLUS_2"


@dataclass(slots=True)
class Keyframe:
    """One saved raw video frame tied to a phase / contact-neighbor role."""

    phase: str
    frame_index: int
    timestamp: float
    file_path: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "file_path": self.file_path,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class KeyframeSet:
    video: str
    output_dir: str
    keyframes: list[Keyframe] = field(default_factory=list)
    notes: str = (
        "Raw frames selected from PhaseSequence indices; "
        "no additional CV inference or OpenAI."
    )

    @property
    def count(self) -> int:
        return len(self.keyframes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "output_dir": self.output_dir,
            "count": self.count,
            "notes": self.notes,
            "keyframes": [kf.to_dict() for kf in self.keyframes],
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
