"""Technique evaluation schemas (calibrated reference-distribution aware)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.schemas.phases import SmashPhase
from app.schemas.provenance import provenance_fields_from_object
from app.schemas.reference import ReferenceEvidence
from app.schemas.technique_calibration import (
    SEVERITY_CALIBRATION_VERSION,
    EvaluationConfidence,
    IssueStatus,
)

# Rule versions for explainability / reproducibility.
TECHNIQUE_RULE_VERSION_REFERENCE = "reference_distribution_v2"
TECHNIQUE_RULE_VERSION_FALLBACK = "provisional_fallback_v1"
TECHNIQUE_RULE_VERSION_LEGACY = "legacy_hardcoded_v1"


class IssueSeverity(str, Enum):
    """Backward-compatible severity (maps from IssueStatus)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def status_to_severity(status: IssueStatus) -> IssueSeverity | None:
    if status == IssueStatus.MINOR:
        return IssueSeverity.LOW
    if status == IssueStatus.MODERATE:
        return IssueSeverity.MEDIUM
    if status in (IssueStatus.MAJOR,):
        return IssueSeverity.HIGH
    return None


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
    # Explainability / calibration fields (C3–C4).
    metric_name: str = ""
    reference_median: float | None = None
    deviation: float | None = None
    percentile_position: float | None = None
    reference_percentile: float | None = None
    measurement_confidence: float = 0.0
    phase_confidence: float = 0.0
    video_quality_confidence: float = 0.0
    reference_confidence: float = 0.0
    combined_confidence: float = 0.0
    rule_version: str = TECHNIQUE_RULE_VERSION_REFERENCE
    severity_calibration_version: str = SEVERITY_CALIBRATION_VERSION
    decision_mode: str = "reference_distribution"
    status: str = IssueStatus.MINOR.value
    uncertain: bool = False
    status_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "issue_code": self.code,
            "phase": self.phase.value,
            "severity": self.severity.value,
            "status": self.status,
            "status_reason": self.status_reason,
            "confidence": self.confidence,
            "measured_value": self.measured_value,
            "unit": self.unit,
            "reference_range": self.reference_range.to_dict(),
            "reference_low": self.reference_range.min,
            "reference_high": self.reference_range.max,
            "description": self.description,
            "reference_profile_id": self.reference_profile_id,
            "metric_name": self.metric_name,
            "reference_median": self.reference_median,
            "deviation": self.deviation,
            "percentile_position": self.percentile_position,
            "reference_percentile": self.reference_percentile
            if self.reference_percentile is not None
            else self.percentile_position,
            "measurement_confidence": self.measurement_confidence,
            "phase_confidence": self.phase_confidence,
            "video_quality_confidence": self.video_quality_confidence,
            "reference_confidence": self.reference_confidence,
            "combined_confidence": self.combined_confidence,
            "rule_version": self.rule_version,
            "severity_calibration_version": self.severity_calibration_version,
            "decision_mode": self.decision_mode,
            "uncertain": self.uncertain,
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
    reference_profile_version: str = ""
    profile_match_level: str = ""
    decision_mode: str = ""
    rule_version: str = ""
    severity_calibration_version: str = SEVERITY_CALIBRATION_VERSION
    evaluation_confidence: EvaluationConfidence | None = None
    analysis_id: str = ""
    snapshot_id: str = ""
    fingerprint: str = ""
    snapshot_schema_version: str = ""
    artifact_schema_version: str = ""
    artifact_role: str = ""

    @property
    def issue_count(self) -> int:
        return len(
            [
                i
                for i in self.issues
                if i.status
                not in (
                    IssueStatus.NO_ISSUE.value,
                    IssueStatus.INSUFFICIENT_EVIDENCE.value,
                )
            ]
        )

    @property
    def insufficient_evidence_count(self) -> int:
        return sum(
            1
            for i in self.issues
            if i.status == IssueStatus.INSUFFICIENT_EVIDENCE.value
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "video": self.video,
            "confidence": self.confidence,
            "issue_count": self.issue_count,
            "insufficient_evidence_count": self.insufficient_evidence_count,
            "reference_profile_id": self.reference_profile_id,
            "reference_profile_version": self.reference_profile_version,
            "profile_match_level": self.profile_match_level,
            "decision_mode": self.decision_mode,
            "rule_version": self.rule_version,
            "severity_calibration_version": self.severity_calibration_version,
            "evaluation_confidence": (
                self.evaluation_confidence.to_dict()
                if self.evaluation_confidence
                else None
            ),
            "issues": [issue.to_dict() for issue in self.issues],
        }
        payload.update(provenance_fields_from_object(self))
        return payload

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
