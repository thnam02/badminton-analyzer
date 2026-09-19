"""C5 legacy vs reference coach-label validation tests."""

from __future__ import annotations

import pytest

from app.processing.technique import evaluate_technique
from app.processing.technique_legacy import evaluate_technique_legacy
from app.schemas.phases import SmashPhase
from app.schemas.stroke_metrics import StrokeMetrics
from app.schemas.technique import (
    IssueSeverity,
    ReferenceRange,
    TechniqueEvaluation,
    TechniqueIssue,
)
from app.schemas.technique_calibration import IssueStatus
from app.validation.technique_ab_comparison import (
    build_validation_decision,
    compare_legacy_vs_reference,
    coverage_and_insufficient_rates,
    severity_agreement_stats,
)
from app.validation.technique_annotations import (
    CoachIssueLabel,
    StrokeTechniqueAnnotation,
    TechniqueValidationAnnotationSet,
)
from app.validation.technique_metrics import ConfusionCounts, metrics_from_confusion
from app.validation.technique_validator import TechniqueIssueValidator, _predicted_codes
from tests.test_technique import _build_pipeline
from tests.test_technique_calibration import _elbow_only_profile


def test_confusion_matrix_calculations() -> None:
    counts = ConfusionCounts()
    counts.add(gt_present=True, pred_present=True)
    counts.add(gt_present=True, pred_present=True)
    counts.add(gt_present=False, pred_present=True)
    counts.add(gt_present=True, pred_present=False)
    counts.add(gt_present=False, pred_present=False)
    metrics = metrics_from_confusion(counts)
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3
    assert metrics["specificity"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 1 / 3
    assert metrics["support"] == 3
    assert metrics["confusion_matrix"]["true_positive"] == 2


def test_insufficient_evidence_excluded_from_positive_predictions() -> None:
    issue = TechniqueIssue(
        code="INSUFFICIENT_ELBOW_EXTENSION",
        phase=SmashPhase.ESTIMATED_CONTACT,
        severity=IssueSeverity.LOW,
        confidence=0.3,
        measured_value=110.0,
        reference_range=ReferenceRange(min=150.0, max=180.0),
        unit="deg",
        status=IssueStatus.INSUFFICIENT_EVIDENCE.value,
    )
    assert _predicted_codes([issue]) == set()
    judged = TechniqueIssue(
        code="INSUFFICIENT_ELBOW_EXTENSION",
        phase=SmashPhase.ESTIMATED_CONTACT,
        severity=IssueSeverity.HIGH,
        confidence=0.9,
        measured_value=110.0,
        reference_range=ReferenceRange(min=150.0, max=180.0),
        unit="deg",
        status=IssueStatus.MAJOR.value,
    )
    assert _predicted_codes([judged]) == {"INSUFFICIENT_ELBOW_EXTENSION"}


def test_coverage_calculations() -> None:
    judged = TechniqueEvaluation(
        video="a.mp4",
        issues=[
            TechniqueIssue(
                code="INSUFFICIENT_ELBOW_EXTENSION",
                phase=SmashPhase.ESTIMATED_CONTACT,
                severity=IssueSeverity.MEDIUM,
                confidence=0.8,
                measured_value=120.0,
                reference_range=ReferenceRange(min=150.0, max=None),
                unit="deg",
                status=IssueStatus.MODERATE.value,
            )
        ],
    )
    refused = TechniqueEvaluation(
        video="b.mp4",
        issues=[
            TechniqueIssue(
                code="INSUFFICIENT_ELBOW_EXTENSION",
                phase=SmashPhase.ESTIMATED_CONTACT,
                severity=IssueSeverity.LOW,
                confidence=0.3,
                measured_value=120.0,
                reference_range=ReferenceRange(min=150.0, max=None),
                unit="deg",
                status=IssueStatus.INSUFFICIENT_EVIDENCE.value,
            )
        ],
    )
    clean = TechniqueEvaluation(video="c.mp4", issues=[])
    coverage, ie_rate, judged_n, total = coverage_and_insufficient_rates(
        {"a": judged, "b": refused, "c": clean}
    )
    assert total == 3
    assert judged_n == 2
    assert coverage == pytest.approx(2 / 3)
    assert ie_rate == pytest.approx(1 / 3)


def test_uncertain_coach_labels_excluded() -> None:
    annotations = TechniqueValidationAnnotationSet(
        strokes=[
            StrokeTechniqueAnnotation(
                stroke_id="s1",
                issues={
                    "INSUFFICIENT_ELBOW_EXTENSION": CoachIssueLabel(label="uncertain"),
                },
                analysis_confidence=0.9,
            )
        ]
    )
    preds = {
        "s1": TechniqueEvaluation(
            video="s1",
            issues=[
                TechniqueIssue(
                    code="INSUFFICIENT_ELBOW_EXTENSION",
                    phase=SmashPhase.ESTIMATED_CONTACT,
                    severity=IssueSeverity.HIGH,
                    confidence=0.9,
                    measured_value=110.0,
                    reference_range=ReferenceRange(min=150.0, max=None),
                    unit="deg",
                    status=IssueStatus.MAJOR.value,
                )
            ],
        )
    }
    report = TechniqueIssueValidator().validate(annotations, preds)
    assert report.uncertain_summary["total_uncertain_labels"] == 1
    metrics = report.per_issue.get("INSUFFICIENT_ELBOW_EXTENSION")
    assert metrics is not None
    assert metrics.sample_count == 0


def test_severity_agreement() -> None:
    stats = severity_agreement_stats(
        [
            ("MODERATE", "MODERATE"),
            ("MODERATE", "MINOR"),
            ("MAJOR", "MINOR"),
            ("HIGH", "MAJOR"),  # alias
            (None, "MINOR"),
        ]
    )
    assert stats["comparable_pairs"] == 4
    assert stats["exact_agreement"] == 2
    assert stats["within_one_level_agreement"] == 3


def test_evaluator_ab_comparison_and_validation_status() -> None:
    _, _, _, _, base_metrics = _build_pipeline()
    profile = _elbow_only_profile()

    def _metrics(elbow: float, phase: float = 0.95) -> StrokeMetrics:
        m = StrokeMetrics(
            video=base_metrics.video,
            phase_confidence=phase,
            estimated_contact_frame_index=base_metrics.estimated_contact_frame_index,
            contact_elbow_angle_deg=elbow,
        )
        return m

    strokes = [
        StrokeTechniqueAnnotation(
            stroke_id="tp1",
            issues={
                "INSUFFICIENT_ELBOW_EXTENSION": CoachIssueLabel(
                    label="present", coach_severity="MAJOR"
                )
            },
            analysis_confidence=0.9,
            camera_view="side_45",
            skill_level="intermediate",
        ),
        StrokeTechniqueAnnotation(
            stroke_id="tn1",
            issues={
                "INSUFFICIENT_ELBOW_EXTENSION": CoachIssueLabel(label="absent")
            },
            analysis_confidence=0.9,
            camera_view="side_45",
            skill_level="intermediate",
        ),
        StrokeTechniqueAnnotation(
            stroke_id="fp_prone",
            issues={
                "INSUFFICIENT_ELBOW_EXTENSION": CoachIssueLabel(label="absent")
            },
            analysis_confidence=0.9,
        ),
    ]
    annotations = TechniqueValidationAnnotationSet(strokes=strokes)
    metrics_by = {
        "tp1": _metrics(110.0),
        "tn1": _metrics(168.0),
        "fp_prone": _metrics(110.0, phase=0.1),  # reference should refuse
    }

    def legacy_predict(metrics: StrokeMetrics, _ann):
        return evaluate_technique_legacy(metrics, quality_confidence=0.9)

    def reference_predict(metrics: StrokeMetrics, _ann):
        return evaluate_technique(
            metrics, profile=profile, quality_confidence=0.9 if metrics.phase_confidence > 0.5 else 0.1
        )

    comparison = compare_legacy_vs_reference(
        annotations,
        metrics_by,
        legacy_predict=legacy_predict,
        reference_predict=reference_predict,
        dataset_version="coach_smash_test_v1",
    )
    assert comparison.legacy.total_stroke_count == 3
    assert comparison.reference.total_stroke_count == 3
    assert "precision" in comparison.legacy.micro_metrics
    assert comparison.reference.insufficient_evidence_rate >= 0.0

    # Small dataset → insufficient_data, not validated.
    decision = build_validation_decision(
        comparison,
        evaluator_version="reference_v1",
        min_labelled_strokes=30,
        min_support_per_issue=5,
    )
    assert decision.validated is False
    assert decision.insufficient_data is True
    assert decision.validation_dataset_version == "coach_smash_test_v1"
