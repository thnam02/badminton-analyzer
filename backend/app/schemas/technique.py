"""Technique evaluation schemas (rule-based V1)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.schemas.phases import SmashPhase
from app.schemas.reference import ReferenceEvidence


class IssueSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True, slots=True)
class ReferenceRange:
    min: float | None = None
    max: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {"min": self.min, "max": self.max}


@dataclass(slots=True)
class TechniqueIssue:
    code: str
    phase: SmashPhase
    severity: IssueSeverity
    confidence: float
    measured_value: float
    reference_range: ReferenceRange
    unit: str
    description: str = ""
    reference_profile_id: str = ""
    reference_evidence: ReferenceEvidence | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "phase": self.phase.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "measured_value": self.measured_value,
            "reference_range": self.reference_range.to_dict(),
            "unit": self.unit,
            "description": self.description,
            "reference_profile_id": self.reference_profile_id,
        }
        if self.reference_evidence is not None:
            payload["reference_evidence"] = self.reference_evidence.to_dict()
        else:
            payload["reference_evidence"] = None
        return payload


@dataclass(slots=True)
class TechniqueEvaluation:
    video: str
    issues: list[TechniqueIssue] = field(default_factory=list)
    confidence: float = 0.0
    reference_profile_id: str = ""

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "confidence": self.confidence,
            "issue_count": self.issue_count,
            "reference_profile_id": self.reference_profile_id,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
