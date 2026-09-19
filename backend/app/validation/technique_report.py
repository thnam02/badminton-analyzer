"""Versioned technique-issue validation report schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TECHNIQUE_VALIDATION_REPORT_VERSION = "1.0.0"


@dataclass(slots=True)
class IssuePredictionOutcome:
    """Per-issue outcome for one stroke (certain labels only contribute to metrics)."""

    issue_code: str
    gt_label: str
    predicted_present: bool
    included_in_metrics: bool
    outcome: str | None
    # outcome ∈ {tp, fp, fn, tn} when included; None when uncertain / excluded.

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_code": self.issue_code,
            "gt_label": self.gt_label,
            "predicted_present": self.predicted_present,
            "included_in_metrics": self.included_in_metrics,
            "outcome": self.outcome,
        }


@dataclass(slots=True)
class StrokeTechniqueResult:
    stroke_id: str
    video: str
    analysis_confidence: float | None
    camera_quality: str | None
    predicted_issue_codes: list[str] = field(default_factory=list)
    outcomes: list[IssuePredictionOutcome] = field(default_factory=list)
    uncertain_label_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stroke_id": self.stroke_id,
            "video": self.video,
            "analysis_confidence": self.analysis_confidence,
            "camera_quality": self.camera_quality,
            "predicted_issue_codes": list(self.predicted_issue_codes),
            "uncertain_label_count": self.uncertain_label_count,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


@dataclass(slots=True)
class IssueClassificationMetrics:
    """Aggregate binary metrics for one issue code (or a group rollup)."""

    issue_code: str
    precision: float | None
    recall: float | None
    f1: float | None
    specificity: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    support: int
    sample_count: int
    confusion_matrix: dict[str, int] = field(default_factory=dict)
    uncertain_excluded_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_code": self.issue_code,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "support": self.support,
            "sample_count": self.sample_count,
            "confusion_matrix": dict(self.confusion_matrix),
            "uncertain_excluded_count": self.uncertain_excluded_count,
        }


@dataclass(slots=True)
class GroupTechniqueBreakdown:
    """Metrics stratified by analysis confidence or camera quality."""

    key: str
    sample_count: int
    per_issue: dict[str, IssueClassificationMetrics] = field(default_factory=dict)
    micro_average: IssueClassificationMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sample_count": self.sample_count,
            "per_issue": {
                k: v.to_dict() for k, v in sorted(self.per_issue.items())
            },
            "micro_average": (
                self.micro_average.to_dict() if self.micro_average else None
            ),
        }


@dataclass(slots=True)
class TechniqueValidationReport:
    """Versioned per-stroke + aggregate technique-issue validation."""

    technique_validation_report_version: str = TECHNIQUE_VALIDATION_REPORT_VERSION
    annotation_version: str = ""
    stroke_count: int = 0
    per_issue: dict[str, IssueClassificationMetrics] = field(default_factory=dict)
    micro_average: IssueClassificationMetrics | None = None
    by_analysis_confidence: list[GroupTechniqueBreakdown] = field(default_factory=list)
    by_camera_quality: list[GroupTechniqueBreakdown] = field(default_factory=list)
    uncertain_summary: dict[str, Any] = field(default_factory=dict)
    strokes: list[StrokeTechniqueResult] = field(default_factory=list)
    notes: str = (
        "Offline technique-issue validation against coach labels. "
        "Does not modify production technique rules, thresholds, or /analyze."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "technique_validation_report_version": (
                self.technique_validation_report_version
            ),
            "annotation_version": self.annotation_version,
            "stroke_count": self.stroke_count,
            "per_issue": {
                k: v.to_dict() for k, v in sorted(self.per_issue.items())
            },
            "micro_average": (
                self.micro_average.to_dict() if self.micro_average else None
            ),
            "by_analysis_confidence": [
                g.to_dict() for g in self.by_analysis_confidence
            ],
            "by_camera_quality": [g.to_dict() for g in self.by_camera_quality],
            "uncertain_summary": dict(self.uncertain_summary),
            "strokes": [s.to_dict() for s in self.strokes],
            "notes": self.notes,
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
