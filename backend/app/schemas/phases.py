"""Smash phase schemas (rule-based V1)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SmashPhase(str, Enum):
    PREPARATION = "PREPARATION"
    BACKSWING = "BACKSWING"
    ACCELERATION = "ACCELERATION"
    # Peak wrist-speed anchor — not true shuttle/racket contact.
    ESTIMATED_CONTACT = "ESTIMATED_CONTACT"
    FOLLOW_THROUGH = "FOLLOW_THROUGH"


@dataclass(slots=True)
class PhaseSegment:
    phase: SmashPhase
    start_frame_index: int
    end_frame_index: int
    start_timestamp: float
    end_timestamp: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class PhaseSequence:
    video: str
    segments: list[PhaseSegment] = field(default_factory=list)
    # Per-frame label for rendering / lookup (frame_index -> phase).
    frame_phases: dict[int, SmashPhase] = field(default_factory=dict)
    estimated_contact_frame_index: int | None = None
    estimated_contact_timestamp: float | None = None
    confidence: float = 0.0
    notes: str = (
        "ESTIMATED_CONTACT is anchored at peak right-wrist speed; "
        "shuttle/racket tracking is not used."
    )

    @property
    def frame_count(self) -> int:
        return len(self.frame_phases)

    def phase_at(self, frame_index: int) -> SmashPhase | None:
        return self.frame_phases.get(frame_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "confidence": self.confidence,
            "estimated_contact_frame_index": self.estimated_contact_frame_index,
            "estimated_contact_timestamp": self.estimated_contact_timestamp,
            "notes": self.notes,
            "segments": [seg.to_dict() for seg in self.segments],
            "frame_phases": {
                str(idx): phase.value for idx, phase in sorted(self.frame_phases.items())
            },
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
