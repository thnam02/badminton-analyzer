"""Versioned coaching validation report schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COACHING_VALIDATION_REPORT_VERSION = "1.0.0"


@dataclass(slots=True)
class AnalysisCoachingValidation:
    """Per-analysis automatic + optional coach-review validation record."""

    analysis_id: str
    video: str
    analysis_confidence: float | None
    faithfulness_findings: list[dict[str, Any]] = field(default_factory=list)
    unsupported_claims: list[dict[str, Any]] = field(default_factory=list)
    faithfulness_claim_count: int = 0
    faithfulness_supported_count: int = 0
    faithfulness_rate: float | None = None
    hallucination_or_unsupported_count: int = 0
    has_hallucination_or_unsupported: bool = False
    priority_issue_codes: list[str] = field(default_factory=list)
    priority_agreement_rate: float | None = None
    coach_ratings: dict[str, int] = field(default_factory=dict)
    coach_comments: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "video": self.video,
            "analysis_confidence": self.analysis_confidence,
            "faithfulness_claim_count": self.faithfulness_claim_count,
            "faithfulness_supported_count": self.faithfulness_supported_count,
            "faithfulness_rate": self.faithfulness_rate,
            "hallucination_or_unsupported_count": (
                self.hallucination_or_unsupported_count
            ),
            "has_hallucination_or_unsupported": self.has_hallucination_or_unsupported,
            "priority_issue_codes": list(self.priority_issue_codes),
            "priority_agreement_rate": self.priority_agreement_rate,
            "coach_ratings": dict(self.coach_ratings),
            "coach_comments": self.coach_comments,
            "faithfulness_findings": list(self.faithfulness_findings),
            "unsupported_claims": list(self.unsupported_claims),
        }


@dataclass(slots=True)
class ConfidenceBinCoachingStats:
    key: str
    sample_count: int
    evidence_faithfulness_rate: float | None
    hallucination_unsupported_rate: float | None
    mean_priority_agreement_rate: float | None
    mean_coach_ratings: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sample_count": self.sample_count,
            "evidence_faithfulness_rate": self.evidence_faithfulness_rate,
            "hallucination_unsupported_rate": self.hallucination_unsupported_rate,
            "mean_priority_agreement_rate": self.mean_priority_agreement_rate,
            "mean_coach_ratings": dict(self.mean_coach_ratings),
        }


@dataclass(slots=True)
class CoachingValidationReport:
    """Versioned aggregate + per-analysis coaching validation export."""

    coaching_validation_report_version: str = COACHING_VALIDATION_REPORT_VERSION
    annotation_version: str = ""
    analysis_count: int = 0
    evidence_faithfulness_rate: float | None = None
    hallucination_unsupported_rate: float | None = None
    mean_priority_agreement_rate: float | None = None
    mean_coach_ratings: dict[str, float | None] = field(default_factory=dict)
    by_analysis_confidence: list[ConfidenceBinCoachingStats] = field(
        default_factory=list
    )
    analyses: list[AnalysisCoachingValidation] = field(default_factory=list)
    notes: str = (
        "Offline OpenAI coaching validation against EvidencePackage + optional "
        "coach review. Does not modify production prompts, models, or /analyze."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "coaching_validation_report_version": (
                self.coaching_validation_report_version
            ),
            "annotation_version": self.annotation_version,
            "analysis_count": self.analysis_count,
            "evidence_faithfulness_rate": self.evidence_faithfulness_rate,
            "hallucination_unsupported_rate": self.hallucination_unsupported_rate,
            "mean_priority_agreement_rate": self.mean_priority_agreement_rate,
            "mean_coach_ratings": dict(self.mean_coach_ratings),
            "by_analysis_confidence": [
                g.to_dict() for g in self.by_analysis_confidence
            ],
            "analyses": [a.to_dict() for a in self.analyses],
            "notes": self.notes,
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
