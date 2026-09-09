"""Resolved contact event schemas (tracked or kinematic fallback)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONTACT_TYPE_TRACKED = "TRACKED_CONTACT"
CONTACT_TYPE_KINEMATIC = "KINEMATIC_ESTIMATE"
# Backward-compatible alias used by older evidence consumers.
CONTACT_TYPE_ESTIMATED = CONTACT_TYPE_KINEMATIC


@dataclass(slots=True)
class ContactSignalEvidence:
    """One scored signal that contributed to contact resolution."""

    name: str
    value: float | None = None
    weight: float = 0.0
    score: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "weight": self.weight,
            "score": self.score,
            "notes": self.notes,
        }


@dataclass(slots=True)
class ContactEvent:
    """Resolved smash contact for metrics / evidence."""

    contact_type: str
    frame_index: int
    timestamp: float
    confidence: float
    evidence: list[ContactSignalEvidence] = field(default_factory=list)
    kinematic_frame_index: int | None = None
    kinematic_timestamp: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact_type": self.contact_type,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "kinematic_frame_index": self.kinematic_frame_index,
            "kinematic_timestamp": self.kinematic_timestamp,
            "notes": self.notes,
            "evidence": [e.to_dict() for e in self.evidence],
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
