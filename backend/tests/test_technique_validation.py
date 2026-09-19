"""Synthetic tests for technique-issue validation metrics and confusion matrices."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.phases import SmashPhase
from app.schemas.technique import (
    IssueSeverity,
    ReferenceRange,
    TechniqueEvaluation,
    TechniqueIssue,
)
from app.validation.technique_annotations import (
    CoachIssueLabel,
    StrokeTechniqueAnnotation,
    TechniqueValidationAnnotationSet,
    load_technique_annotation_set,
)
from app.validation.technique_issues import (
    ISSUE_INSUFFICIENT_ELBOW_EXTENSION,
    ISSUE_LOW_CONTACT_POSTURE,
    ISSUE_LOW_KNEE_CONTRIBUTION,
    ISSUE_POOR_ARM_ACCELERATION_TIMING,
    ISSUE_WEAK_FOLLOW_THROUGH,
    LABEL_ABSENT,
    LABEL_PRESENT,
    LABEL_UNCERTAIN,
)
from app.validation.technique_metrics import (
    ConfusionCounts,
    f1_score,
    false_negative_rate,
    false_positive_rate,
    metrics_from_confusion,
    precision,
    recall,
    specificity,
)
from app.validation.technique_report import TECHNIQUE_VALIDATION_REPORT_VERSION
from app.validation.technique_validator import (
    TechniqueIssueValidator,
    export_technique_validation_report,
    validate_technique_issues,
)


def _issue(code: str) -> TechniqueIssue:
    return TechniqueIssue(
        code=code,
        phase=SmashPhase.ACCELERATION,
        severity=IssueSeverity.MEDIUM,
        confidence=0.8,
        measured_value=1.0,
        reference_range=ReferenceRange(min=0.0, max=2.0),
        unit="unit",
        description=code,
    )


def _stroke(
    stroke_id: str,
    issues: dict[str, str],
    *,
    conf: float | None = 0.8,
    camera: str | None = "good",
) -> StrokeTechniqueAnnotation:
    return StrokeTechniqueAnnotation(
        stroke_id=stroke_id,
        video=f"{stroke_id}.mp4",
        issues={
            code: CoachIssueLabel(label=label) for code, label in issues.items()
        },
        analysis_confidence=conf,
        camera_quality=camera,
    )


def test_confusion_and_metric_helpers_exact() -> None:
    """Known TP/FP/FN/TN → exact precision, recall, F1, specificity, FPR, FNR."""
    # TP=2, FP=1, FN=1, TN=6
    counts = ConfusionCounts(
        true_positive=2,
        false_positive=1,
        false_negative=1,
        true_negative=6,
    )
    assert counts.support == 3
    assert counts.total == 10

    assert precision(2, 1) == pytest.approx(2 / 3)
    assert recall(2, 1) == pytest.approx(2 / 3)
    assert specificity(6, 1) == pytest.approx(6 / 7)
    assert f1_score(2 / 3, 2 / 3) == pytest.approx(2 / 3)
    assert false_positive_rate(1, 6) == pytest.approx(1 / 7)
    assert false_negative_rate(1, 2) == pytest.approx(1 / 3)

    metrics = metrics_from_confusion(counts)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["specificity"] == pytest.approx(6 / 7)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 7)
    assert metrics["false_negative_rate"] == pytest.approx(1 / 3)
    assert metrics["support"] == 3
    assert metrics["sample_count"] == 10
    assert metrics["confusion_matrix"]["true_positive"] == 2
    assert metrics["confusion_matrix"]["false_positive"] == 1
    assert metrics["confusion_matrix"]["false_negative"] == 1
    assert metrics["confusion_matrix"]["true_negative"] == 6


def test_confusion_counts_add_all_cells() -> None:
    counts = ConfusionCounts()
    counts.add(gt_present=True, pred_present=True)  # TP
    counts.add(gt_present=False, pred_present=True)  # FP
    counts.add(gt_present=True, pred_present=False)  # FN
    counts.add(gt_present=False, pred_present=False)  # TN
    assert counts.to_dict() == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "support": 2,
        "total": 4,
    }


def test_empty_denominators_return_none() -> None:
    assert precision(0, 0) is None
    assert recall(0, 0) is None
    assert specificity(0, 0) is None
    assert f1_score(None, 0.5) is None
    assert false_positive_rate(0, 0) is None
    assert false_negative_rate(0, 0) is None
    empty = metrics_from_confusion(ConfusionCounts())
    assert empty["precision"] is None
    assert empty["sample_count"] == 0


def test_validate_exact_confusion_for_one_issue() -> None:
    """
    Four strokes for INSUFFICIENT_ELBOW_EXTENSION:
      present+pred → TP
      absent+pred  → FP
      present-miss → FN
      absent-miss  → TN
    """
    code = ISSUE_INSUFFICIENT_ELBOW_EXTENSION
    ann = TechniqueValidationAnnotationSet(
        strokes=[
            _stroke("s_tp", {code: LABEL_PRESENT}),
            _stroke("s_fp", {code: LABEL_ABSENT}),
            _stroke("s_fn", {code: LABEL_PRESENT}),
            _stroke("s_tn", {code: LABEL_ABSENT}),
        ]
    )
    preds = {
        "s_tp": [_issue(code)],
        "s_fp": [_issue(code)],
        "s_fn": [],
        "s_tn": [],
    }
    report = validate_technique_issues(ann, preds)
    m = report.per_issue[code]
    assert m.confusion_matrix == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "support": 2,
        "total": 4,
    }
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(0.5)
    assert m.specificity == pytest.approx(0.5)
    assert m.false_positive_rate == pytest.approx(0.5)
    assert m.false_negative_rate == pytest.approx(0.5)
    assert m.support == 2
    assert m.sample_count == 4


def test_uncertain_labels_excluded_and_reported_separately() -> None:
    code = ISSUE_LOW_KNEE_CONTRIBUTION
    ann = TechniqueValidationAnnotationSet(
        strokes=[
            _stroke("a", {code: LABEL_PRESENT}),
            _stroke("b", {code: LABEL_UNCERTAIN}),
            _stroke("c", {code: LABEL_ABSENT}),
        ]
    )
    preds = {
        "a": [_issue(code)],
        "b": [_issue(code)],  # prediction present but GT uncertain → excluded
        "c": [],
    }
    report = validate_technique_issues(ann, preds)
    m = report.per_issue[code]
    assert m.sample_count == 2  # only present + absent
    assert m.confusion_matrix["true_positive"] == 1
    assert m.confusion_matrix["true_negative"] == 1
    assert m.uncertain_excluded_count == 1

    summary = report.uncertain_summary
    assert summary["excluded_from_primary_metrics"] is True
    assert summary["total_uncertain_labels"] == 1
    assert summary["by_issue"][code]["uncertain_count"] == 1
    assert summary["by_issue"][code]["predicted_present"] == 1

    uncertain_outcome = next(
        o
        for s in report.strokes
        if s.stroke_id == "b"
        for o in s.outcomes
        if o.issue_code == code
    )
    assert uncertain_outcome.included_in_metrics is False
    assert uncertain_outcome.outcome is None


def test_breakdown_by_confidence_and_camera_quality() -> None:
    code = ISSUE_WEAK_FOLLOW_THROUGH
    ann = TechniqueValidationAnnotationSet(
        strokes=[
            _stroke("hi_good", {code: LABEL_PRESENT}, conf=0.9, camera="good"),
            _stroke("lo_poor", {code: LABEL_PRESENT}, conf=0.3, camera="poor"),
            _stroke("hi_poor", {code: LABEL_ABSENT}, conf=0.85, camera="poor"),
        ]
    )
    # Correct on high/good; miss on low/poor; false alarm on high/poor
    preds = {
        "hi_good": [_issue(code)],
        "lo_poor": [],
        "hi_poor": [_issue(code)],
    }
    report = validate_technique_issues(ann, preds)

    by_conf = {g.key: g for g in report.by_analysis_confidence}
    assert by_conf["high_>=0.75"].sample_count == 2
    assert by_conf["low_<0.5"].sample_count == 1
    assert by_conf["low_<0.5"].per_issue[code].recall == pytest.approx(0.0)
    assert by_conf["high_>=0.75"].per_issue[code].precision == pytest.approx(0.5)

    by_cam = {g.key: g for g in report.by_camera_quality}
    assert set(by_cam) == {"good", "poor"}
    assert by_cam["good"].per_issue[code].f1 == pytest.approx(1.0)
    assert by_cam["poor"].per_issue[code].false_positive_rate == pytest.approx(1.0)


def test_accepts_technique_evaluation_and_exports_report(tmp_path: Path) -> None:
    codes = [
        ISSUE_INSUFFICIENT_ELBOW_EXTENSION,
        ISSUE_LOW_CONTACT_POSTURE,
        ISSUE_POOR_ARM_ACCELERATION_TIMING,
    ]
    ann = TechniqueValidationAnnotationSet(
        strokes=[
            _stroke(
                "clip",
                {
                    ISSUE_INSUFFICIENT_ELBOW_EXTENSION: LABEL_PRESENT,
                    ISSUE_LOW_CONTACT_POSTURE: LABEL_ABSENT,
                    ISSUE_POOR_ARM_ACCELERATION_TIMING: LABEL_PRESENT,
                },
            )
        ]
    )
    evaluation = TechniqueEvaluation(
        video="clip.mp4",
        issues=[
            _issue(ISSUE_INSUFFICIENT_ELBOW_EXTENSION),
            # LOW_CONTACT_POSTURE absent in pred → TN
            # POOR_ARM_ACCELERATION_TIMING missing → FN
        ],
        confidence=0.7,
    )
    report = TechniqueIssueValidator().validate(ann, {"clip": evaluation})
    assert report.per_issue[ISSUE_INSUFFICIENT_ELBOW_EXTENSION].precision == 1.0
    assert report.per_issue[ISSUE_LOW_CONTACT_POSTURE].specificity == 1.0
    assert report.per_issue[ISSUE_POOR_ARM_ACCELERATION_TIMING].recall == 0.0

    out = tmp_path / "technique_validation.json"
    export_technique_validation_report(report, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (
        data["technique_validation_report_version"]
        == TECHNIQUE_VALIDATION_REPORT_VERSION
    )
    assert data["stroke_count"] == 1
    assert data["strokes"][0]["predicted_issue_codes"] == [
        ISSUE_INSUFFICIENT_ELBOW_EXTENSION
    ]
    assert "per_issue" in data
    assert "by_analysis_confidence" in data
    assert "by_camera_quality" in data
    assert "uncertain_summary" in data
    for code in codes:
        assert code in data["per_issue"]


def test_load_example_technique_annotations() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "pose_validation"
        / "example_technique_annotations.json"
    )
    loaded = load_technique_annotation_set(path)
    assert len(loaded.strokes) == 1
    stroke = loaded.strokes[0]
    assert stroke.issues[ISSUE_INSUFFICIENT_ELBOW_EXTENSION].label == LABEL_PRESENT
    assert stroke.issues[ISSUE_POOR_ARM_ACCELERATION_TIMING].label == LABEL_UNCERTAIN
    assert stroke.camera_quality == "good"
    assert stroke.analysis_confidence == pytest.approx(0.85)


def test_invalid_coach_label_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid coach label"):
        CoachIssueLabel(label="maybe")
