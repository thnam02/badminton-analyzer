"""Legacy vs reference technique-evaluator A/B comparison (C5).

Runs both evaluators on the same coach-labelled strokes and produces a
comparison report plus an explicit validation decision artifact. Does **not**
auto-promote the reference evaluator to production.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.schemas.stroke_metrics import StrokeMetrics
from app.schemas.technique import TechniqueEvaluation, TechniqueIssue
from app.schemas.technique_calibration import IssueStatus
from app.validation.technique_annotations import (
    StrokeTechniqueAnnotation,
    TechniqueValidationAnnotationSet,
)
from app.validation.technique_metrics import ConfusionCounts, metrics_from_confusion
from app.validation.technique_report import TechniqueValidationReport
from app.validation.technique_validator import TechniqueIssueValidator

SEVERITY_ORDER = ("MINOR", "MODERATE", "MAJOR")
_SEVERITY_INDEX = {s: i for i, s in enumerate(SEVERITY_ORDER)}

# Map legacy IssueSeverity / status strings into the calibrated ordinal scale.
_SEVERITY_ALIASES = {
    "LOW": "MINOR",
    "MEDIUM": "MODERATE",
    "HIGH": "MAJOR",
    "MINOR": "MINOR",
    "MODERATE": "MODERATE",
    "MAJOR": "MAJOR",
}


@dataclass(slots=True)
class EvaluatorRunSummary:
    """Aggregate metrics for one evaluator on a shared validation set."""

    evaluator_id: str
    report: TechniqueValidationReport
    coverage_rate: float
    insufficient_evidence_rate: float
    judged_stroke_count: int
    total_stroke_count: int
    micro_metrics: dict[str, Any] = field(default_factory=dict)
    per_issue: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_camera_view: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_skill_level: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_analysis_confidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_pose_confidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_contact_confidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_reference_profile: dict[str, dict[str, Any]] = field(default_factory=dict)
    severity_agreement: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "coverage_rate": self.coverage_rate,
            "insufficient_evidence_rate": self.insufficient_evidence_rate,
            "judged_stroke_count": self.judged_stroke_count,
            "total_stroke_count": self.total_stroke_count,
            "micro_metrics": dict(self.micro_metrics),
            "per_issue": {k: dict(v) for k, v in self.per_issue.items()},
            "by_camera_view": dict(self.by_camera_view),
            "by_skill_level": dict(self.by_skill_level),
            "by_analysis_confidence": dict(self.by_analysis_confidence),
            "by_pose_confidence": dict(self.by_pose_confidence),
            "by_contact_confidence": dict(self.by_contact_confidence),
            "by_reference_profile": dict(self.by_reference_profile),
            "severity_agreement": dict(self.severity_agreement),
            "report": self.report.to_dict(),
        }


@dataclass(slots=True)
class TechniqueABComparisonReport:
    """Side-by-side legacy vs reference validation report."""

    dataset_version: str
    legacy: EvaluatorRunSummary
    reference: EvaluatorRunSummary
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "notes": self.notes,
            "legacy": self.legacy.to_dict(),
            "reference": self.reference.to_dict(),
            "headline": {
                "legacy": {
                    "precision": self.legacy.micro_metrics.get("precision"),
                    "recall": self.legacy.micro_metrics.get("recall"),
                    "f1": self.legacy.micro_metrics.get("f1"),
                    "coverage": self.legacy.coverage_rate,
                    "insufficient_evidence_rate": self.legacy.insufficient_evidence_rate,
                },
                "reference": {
                    "precision": self.reference.micro_metrics.get("precision"),
                    "recall": self.reference.micro_metrics.get("recall"),
                    "f1": self.reference.micro_metrics.get("f1"),
                    "coverage": self.reference.coverage_rate,
                    "insufficient_evidence_rate": self.reference.insufficient_evidence_rate,
                },
            },
        }

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


@dataclass(slots=True)
class ValidationDecisionArtifact:
    """Explicit gate before promoting a reference evaluator configuration."""

    evaluator_version: str
    validated: bool
    validation_dataset_version: str
    notes: str
    metrics: dict[str, Any] = field(default_factory=dict)
    insufficient_data: bool = False
    min_labelled_strokes: int = 30
    min_support_per_issue: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_version": self.evaluator_version,
            "validated": self.validated,
            "validation_dataset_version": self.validation_dataset_version,
            "notes": self.notes,
            "insufficient_data": self.insufficient_data,
            "min_labelled_strokes": self.min_labelled_strokes,
            "min_support_per_issue": self.min_support_per_issue,
            "metrics": dict(self.metrics),
        }

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


PredictFn = Callable[[StrokeMetrics, StrokeTechniqueAnnotation], TechniqueEvaluation]


def compare_legacy_vs_reference(
    annotations: TechniqueValidationAnnotationSet,
    metrics_by_stroke: Mapping[str, StrokeMetrics],
    *,
    legacy_predict: PredictFn,
    reference_predict: PredictFn,
    dataset_version: str = "coach_smash_v1",
    notes: str = "",
) -> TechniqueABComparisonReport:
    """Run both evaluators on the same samples and compare coach-label metrics."""
    legacy_preds: dict[str, TechniqueEvaluation] = {}
    reference_preds: dict[str, TechniqueEvaluation] = {}
    for ann in annotations.strokes:
        metrics = metrics_by_stroke.get(ann.stroke_id)
        if metrics is None and ann.video:
            metrics = metrics_by_stroke.get(ann.video)
        if metrics is None:
            continue
        legacy_preds[ann.stroke_id] = legacy_predict(metrics, ann)
        reference_preds[ann.stroke_id] = reference_predict(metrics, ann)

    validator = TechniqueIssueValidator()
    legacy_report = validator.validate(annotations, legacy_preds)
    reference_report = validator.validate(annotations, reference_preds)

    return TechniqueABComparisonReport(
        dataset_version=dataset_version,
        legacy=_summarize_run(
            "legacy_hardcoded_v1",
            legacy_report,
            legacy_preds,
            annotations,
        ),
        reference=_summarize_run(
            "reference_distribution_v2",
            reference_report,
            reference_preds,
            annotations,
        ),
        notes=notes,
    )


def build_validation_decision(
    comparison: TechniqueABComparisonReport,
    *,
    evaluator_version: str = "reference_v1",
    min_labelled_strokes: int = 30,
    min_support_per_issue: int = 5,
    min_f1: float = 0.70,
    min_precision: float = 0.75,
    require_coverage_at_least: float = 0.70,
) -> ValidationDecisionArtifact:
    """Decide whether a reference configuration may be marked validated.

    Never auto-promotes solely because the reference evaluator exists. Marks
    ``insufficient_data`` clearly when the coach set is too small.
    """
    ref = comparison.reference
    stroke_n = ref.total_stroke_count
    f1 = ref.micro_metrics.get("f1")
    precision = ref.micro_metrics.get("precision")
    supports = [
        int(m.get("support") or 0)
        for m in ref.per_issue.values()
    ]
    thin_support = any(s < min_support_per_issue for s in supports) if supports else True

    insufficient = stroke_n < min_labelled_strokes or thin_support
    notes_parts: list[str] = []
    if insufficient:
        notes_parts.append(
            f"Insufficient validation data "
            f"(strokes={stroke_n}, min={min_labelled_strokes}; "
            f"thin_issue_support={thin_support})."
        )
        return ValidationDecisionArtifact(
            evaluator_version=evaluator_version,
            validated=False,
            validation_dataset_version=comparison.dataset_version,
            notes=" ".join(notes_parts),
            metrics=comparison.to_dict()["headline"],
            insufficient_data=True,
            min_labelled_strokes=min_labelled_strokes,
            min_support_per_issue=min_support_per_issue,
        )

    ok = True
    if f1 is None or float(f1) < min_f1:
        ok = False
        notes_parts.append(f"F1 {f1} below minimum {min_f1}.")
    if precision is None or float(precision) < min_precision:
        ok = False
        notes_parts.append(f"Precision {precision} below minimum {min_precision}.")
    if ref.coverage_rate < require_coverage_at_least:
        ok = False
        notes_parts.append(
            f"Coverage {ref.coverage_rate:.3f} below minimum {require_coverage_at_least}."
        )
    if ok:
        notes_parts.append(
            "Reference evaluator met calibrated gates on this dataset; "
            "still requires explicit human promotion for production default."
        )
    else:
        notes_parts.append("Reference evaluator did not meet validation gates.")

    return ValidationDecisionArtifact(
        evaluator_version=evaluator_version,
        validated=ok,
        validation_dataset_version=comparison.dataset_version,
        notes=" ".join(notes_parts),
        metrics=comparison.to_dict()["headline"],
        insufficient_data=False,
        min_labelled_strokes=min_labelled_strokes,
        min_support_per_issue=min_support_per_issue,
    )


def severity_agreement_stats(
    pairs: Sequence[tuple[str | None, str | None]],
) -> dict[str, Any]:
    """Compute exact and within-one-level severity agreement.

    Each pair is ``(coach_severity, system_severity)``. Unknown / missing
    severities are excluded from the denominator.
    """
    comparable = 0
    exact = 0
    within_one = 0
    for coach, system in pairs:
        c = _normalize_severity(coach)
        s = _normalize_severity(system)
        if c is None or s is None:
            continue
        comparable += 1
        if c == s:
            exact += 1
            within_one += 1
            continue
        if abs(_SEVERITY_INDEX[c] - _SEVERITY_INDEX[s]) <= 1:
            within_one += 1
    return {
        "comparable_pairs": comparable,
        "exact_agreement": exact,
        "within_one_level_agreement": within_one,
        "exact_agreement_rate": (exact / comparable) if comparable else None,
        "within_one_level_agreement_rate": (
            within_one / comparable if comparable else None
        ),
    }


def coverage_and_insufficient_rates(
    evaluations: Mapping[str, TechniqueEvaluation],
) -> tuple[float, float, int, int]:
    """Return (coverage, insufficient_rate, judged_count, total)."""
    total = len(evaluations)
    if total == 0:
        return 0.0, 0.0, 0, 0
    insufficient_strokes = 0
    judged = 0
    for evaluation in evaluations.values():
        statuses = [i.status for i in evaluation.issues]
        has_judged = any(
            s
            in (
                IssueStatus.MINOR.value,
                IssueStatus.MODERATE.value,
                IssueStatus.MAJOR.value,
            )
            for s in statuses
        )
        has_ie = any(s == IssueStatus.INSUFFICIENT_EVIDENCE.value for s in statuses)
        if has_judged or not has_ie:
            judged += 1
        if has_ie and not has_judged:
            insufficient_strokes += 1
    coverage = judged / total
    ie_rate = insufficient_strokes / total
    return float(coverage), float(ie_rate), judged, total


def _summarize_run(
    evaluator_id: str,
    report: TechniqueValidationReport,
    predictions: Mapping[str, TechniqueEvaluation],
    annotations: TechniqueValidationAnnotationSet,
) -> EvaluatorRunSummary:
    coverage, ie_rate, judged, total = coverage_and_insufficient_rates(predictions)
    micro = report.micro_average.to_dict() if report.micro_average else {}
    per_issue = {k: v.to_dict() for k, v in report.per_issue.items()}

    ann_by_id = annotations.by_stroke_id()
    severity_pairs: list[tuple[str | None, str | None]] = []
    for stroke_id, evaluation in predictions.items():
        ann = ann_by_id.get(stroke_id)
        if ann is None:
            continue
        for issue in evaluation.issues:
            if issue.status == IssueStatus.INSUFFICIENT_EVIDENCE.value:
                continue
            coach_label = ann.issues.get(issue.code)
            if coach_label is None or not coach_label.is_present:
                continue
            coach_sev = getattr(coach_label, "coach_severity", None)
            system_sev = issue.status if issue.status in _SEVERITY_INDEX else (
                _SEVERITY_ALIASES.get(issue.severity.value)
            )
            severity_pairs.append((coach_sev, system_sev))

    return EvaluatorRunSummary(
        evaluator_id=evaluator_id,
        report=report,
        coverage_rate=coverage,
        insufficient_evidence_rate=ie_rate,
        judged_stroke_count=judged,
        total_stroke_count=total,
        micro_metrics=micro,
        per_issue=per_issue,
        by_camera_view=_stratify_micro(
            annotations, predictions, key_fn=lambda a: getattr(a, "camera_view", None)
        ),
        by_skill_level=_stratify_micro(
            annotations, predictions, key_fn=lambda a: getattr(a, "skill_level", None)
        ),
        by_analysis_confidence={
            g.key: {
                "sample_count": g.sample_count,
                "micro": g.micro_average.to_dict() if g.micro_average else None,
            }
            for g in report.by_analysis_confidence
        },
        by_pose_confidence=_stratify_micro(
            annotations,
            predictions,
            key_fn=lambda a: _bin_confidence(getattr(a, "pose_confidence", None)),
        ),
        by_contact_confidence=_stratify_micro(
            annotations,
            predictions,
            key_fn=lambda a: _bin_confidence(getattr(a, "contact_confidence", None)),
        ),
        by_reference_profile=_stratify_micro(
            annotations,
            predictions,
            key_fn=lambda a: getattr(a, "reference_profile_id", None),
            pred_key_fn=lambda e: e.reference_profile_id or None,
        ),
        severity_agreement=severity_agreement_stats(severity_pairs),
    )


def _stratify_micro(
    annotations: TechniqueValidationAnnotationSet,
    predictions: Mapping[str, TechniqueEvaluation],
    *,
    key_fn: Callable[[StrokeTechniqueAnnotation], str | None],
    pred_key_fn: Callable[[TechniqueEvaluation], str | None] | None = None,
    min_samples: int = 3,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[tuple[StrokeTechniqueAnnotation, TechniqueEvaluation]]] = (
        defaultdict(list)
    )
    for ann in annotations.strokes:
        pred = predictions.get(ann.stroke_id)
        if pred is None:
            continue
        key = key_fn(ann)
        if key is None and pred_key_fn is not None:
            key = pred_key_fn(pred)
        if key is None or str(key).strip() == "":
            continue
        buckets[str(key)].append((ann, pred))

    out: dict[str, dict[str, Any]] = {}
    validator = TechniqueIssueValidator()
    for key, pairs in sorted(buckets.items()):
        if len(pairs) < min_samples:
            continue
        subset_ann = TechniqueValidationAnnotationSet(
            strokes=[p[0] for p in pairs],
            annotation_version=annotations.annotation_version,
        )
        subset_preds = {p[0].stroke_id: p[1] for p in pairs}
        report = validator.validate(subset_ann, subset_preds)
        out[key] = {
            "sample_count": len(pairs),
            "micro": report.micro_average.to_dict() if report.micro_average else None,
        }
    return out


def _bin_confidence(value: float | None) -> str | None:
    if value is None:
        return None
    v = float(value)
    if v < 0.5:
        return "low_<0.5"
    if v < 0.75:
        return "mid_0.5_0.75"
    return "high_>=0.75"


def _normalize_severity(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().upper()
    return _SEVERITY_ALIASES.get(raw)
