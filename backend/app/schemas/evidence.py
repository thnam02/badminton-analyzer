"""Versioned coaching evidence package (assembled from deterministic analysis)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Bump when the coaching-facing evidence shape changes.
EVIDENCE_VERSION = "1.0.0"

STROKE_TYPE_SMASH = "SMASH"
CONTACT_TYPE_ESTIMATED = "ESTIMATED_CONTACT"


@dataclass(slots=True)
class ContactEvidence:
    """Contact event summary for the coaching layer."""

    contact_type: str = CONTACT_TYPE_ESTIMATED
    confidence: float = 0.0
    frame_index: int | None = None
    timestamp: float | None = None
    notes: str = (
        "ESTIMATED_CONTACT is anchored at peak right-wrist speed; "
        "shuttle/racket tracking is not used."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact_type": self.contact_type,
            "confidence": self.confidence,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


@dataclass(slots=True)
class EvidencePackage:
    """Deterministic analysis snapshot consumed by a later coaching layer.

    Does not embed full raw pose sequences.
    """

    evidence_version: str
    video: str
    stroke_type: str
    handedness: str | None
    analysis_confidence: float
    video_quality: dict[str, Any]
    phase_boundaries: list[dict[str, Any]]
    phase_confidence: float
    contact: ContactEvidence
    metrics: dict[str, Any]
    technique_issues: list[dict[str, Any]]
    technique_confidence: float
    keyframes: list[dict[str, Any]]
    keyframes_output_dir: str | None = None
    notes: str = (
        "Assembled from already-computed analysis artifacts; "
        "no biomechanics or CV recalculation."
    )
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evidence_version": self.evidence_version,
            "video": self.video,
            "stroke_type": self.stroke_type,
            "handedness": self.handedness,
            "analysis_confidence": self.analysis_confidence,
            "video_quality": self.video_quality,
            "phase_boundaries": self.phase_boundaries,
            "phase_confidence": self.phase_confidence,
            "contact": self.contact.to_dict(),
            "metrics": self.metrics,
            "technique_issues": self.technique_issues,
            "technique_confidence": self.technique_confidence,
            "keyframes": self.keyframes,
            "keyframes_output_dir": self.keyframes_output_dir,
            "notes": self.notes,
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
