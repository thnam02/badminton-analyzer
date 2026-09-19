"""Deterministic tests for offline OpenAI coaching validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.coaching import (
    CoachingReport,
    CoachingStatus,
    DrillSuggestion,
    PrioritizedIssue,
    Strength,
)
from app.schemas.evidence import (
    EVIDENCE_VERSION,
    STROKE_TYPE_SMASH,
    ContactEvidence,
    EvidencePackage,
)
from app.schemas.video_quality import VideoQualityMetrics, VideoQualityReport
from app.validation.coaching_annotations import (
    CoachReviewAnnotation,
    CoachingValidationAnnotationSet,
    load_coaching_annotation_set,
)
from app.validation.coaching_unsupported import detect_unsupported_claims
from app.validation.coaching_validation_report import (
    COACHING_VALIDATION_REPORT_VERSION,
)
from app.validation.coaching_validator import (
    CoachingValidator,
    export_coaching_validation_report,
    validate_coaching_reports,
)


def _evidence(*, analysis_confidence: float = 0.8) -> EvidencePackage:
    return EvidencePackage(
        evidence_version=EVIDENCE_VERSION,
        video="smash.mp4",
        stroke_type=STROKE_TYPE_SMASH,
        handedness=None,
        analysis_confidence=analysis_confidence,
        video_quality=VideoQualityReport(
            video="smash.mp4",
            usable=True,
            analysis_confidence=analysis_confidence,
            metrics=VideoQualityMetrics(fps=30.0, width=1280, height=720),
        ).to_dict(),
        phase_boundaries=[
            {
                "phase": "ACCELERATION",
                "start_frame_index": 19,
                "end_frame_index": 27,
                "confidence": 0.8,
            },
            {
                "phase": "ESTIMATED_CONTACT",
                "start_frame_index": 28,
                "end_frame_index": 28,
                "confidence": 0.9,
            },
        ],
        phase_confidence=0.75,
        contact=ContactEvidence(
            confidence=0.9,
            frame_index=28,
            timestamp=1.4,
        ),
        metrics={
            "contact_elbow_angle_deg": 120.0,
            "knee_contribution_deg": 5.0,
            "peak_wrist_speed": 2.0,
        },
        technique_issues=[
            {
                "code": "INSUFFICIENT_ELBOW_EXTENSION",
                "phase": "ESTIMATED_CONTACT",
                "severity": "HIGH",
                "confidence": 0.8,
                "measured_value": 120.0,
                "reference_range": {"min": 150.0, "max": 180.0},
                "unit": "deg",
                "description": "Right elbow is not sufficiently extended at estimated contact.",
            },
            {
                "code": "LOW_KNEE_CONTRIBUTION",
                "phase": "ACCELERATION",
                "severity": "MEDIUM",
                "confidence": 0.7,
                "measured_value": 5.0,
                "reference_range": {"min": 12.0, "max": None},
                "unit": "deg",
                "description": "Limited knee extension from preparation to contact.",
            },
        ],
        technique_confidence=0.7,
        keyframes=[],
        keyframes_output_dir=None,
    )


def _correct_report() -> CoachingReport:
    return CoachingReport(
        status=CoachingStatus.OK.value,
        summary=(
            "Main limiter is INSUFFICIENT_ELBOW_EXTENSION at ESTIMATED_CONTACT: "
            "contact elbow is 120 deg vs a 150 deg reference, with confidence 0.8."
        ),
        prioritized_issues=[
            PrioritizedIssue(
                issue_code="INSUFFICIENT_ELBOW_EXTENSION",
                priority=1,
                explanation=(
                    "Evidence shows contact_elbow_angle_deg of 120° at frame 28 "
                    "during ESTIMATED_CONTACT."
                ),
                related_metric_hints=["contact_elbow_angle_deg"],
            )
        ],
        strengths=[
            Strength(
                description="Peak wrist speed of 2.0 is present in the metrics.",
                evidence_refs=["metrics.peak_wrist_speed"],
            )
        ],
        drills=[
            DrillSuggestion(
                name="Shadow smash extension",
                description="Slow shadow smashes focusing on full elbow extension.",
                targets_issue_codes=["INSUFFICIENT_ELBOW_EXTENSION"],
            )
        ],
        caveats=[
            "Contact uses KINEMATIC_ESTIMATE anchored at peak wrist speed."
        ],
        video="smash.mp4",
        evidence_version=EVIDENCE_VERSION,
    )


def _contradictory_report() -> CoachingReport:
    return CoachingReport(
        status=CoachingStatus.OK.value,
        summary=(
            "Contact elbow is 95 deg at frame 99 with confidence 0.2. "
            "Also HIGH_SHOULDER_TILT during BACKSWING. "
            "High muscle activation and injury risk from joint torque; "
            "exact racket speed of 320 km/h."
        ),
        prioritized_issues=[
            PrioritizedIssue(
                issue_code="HIGH_SHOULDER_TILT",
                priority=1,
                explanation="Invented issue with ground-reaction force concerns.",
                related_metric_hints=["shoulder_tilt_deg"],
            )
        ],
        strengths=[],
        drills=[
            DrillSuggestion(
                name="Force-plate loading",
                description="Train against ground-reaction force asymmetry.",
                targets_issue_codes=["HIGH_SHOULDER_TILT"],
            )
        ],
        caveats=[],
        video="smash.mp4",
        evidence_version=EVIDENCE_VERSION,
    )


def test_unsupported_claim_detector_categories() -> None:
    claims = detect_unsupported_claims(
        [
            ("summary", "Elevated muscle activation and joint torque."),
            ("caveat", "Raises injury risk; ground-reaction force is uneven."),
            ("drill", "Exact racket speed of 280 km/h is too low."),
        ]
    )
    categories = {c.category for c in claims}
    assert "muscle_activation" in categories
    assert "joint_force_torque" in categories
    assert "injury_risk" in categories
    assert "ground_reaction_force" in categories
    assert "exact_racket_speed" in categories


def test_correct_report_is_faithful_without_unsupported() -> None:
    record = CoachingValidator().validate_analysis(
        analysis_id="ok",
        report=_correct_report(),
        evidence=_evidence(),
    )
    assert record.has_hallucination_or_unsupported is False
    assert record.unsupported_claims == []
    assert record.faithfulness_rate == pytest.approx(1.0)
    assert record.priority_agreement_rate == pytest.approx(1.0)
    mismatched = [
        f for f in record.faithfulness_findings if not f["supported"]
    ]
    assert mismatched == []


def test_contradictory_report_flags_mismatches_and_unsupported() -> None:
    record = CoachingValidator().validate_analysis(
        analysis_id="bad",
        report=_contradictory_report(),
        evidence=_evidence(),
    )
    assert record.has_hallucination_or_unsupported is True
    assert record.hallucination_or_unsupported_count > 0
    assert record.faithfulness_rate is not None
    assert record.faithfulness_rate < 1.0

    kinds = {f["kind"] for f in record.faithfulness_findings if not f["supported"]}
    assert "measurement" in kinds
    assert "issue_code" in kinds
    assert "contact_frame" in kinds or "confidence" in kinds
    assert "metric_hint" in kinds
    assert "phase" in kinds  # BACKSWING not in evidence phases

    unsupported_cats = {u["category"] for u in record.unsupported_claims}
    assert "muscle_activation" in unsupported_cats
    assert "injury_risk" in unsupported_cats
    assert "exact_racket_speed" in unsupported_cats
    assert "ground_reaction_force" in unsupported_cats


def test_aggregate_rates_and_confidence_bins(tmp_path: Path) -> None:
    reviews = CoachingValidationAnnotationSet(
        reviews=[
            CoachReviewAnnotation(
                analysis_id="ok",
                video="smash.mp4",
                priority_agreement=5,
                technical_correctness=5,
                actionability=4,
                clarity=5,
                drill_relevance=4,
                expected_priority_issue_codes=["INSUFFICIENT_ELBOW_EXTENSION"],
                analysis_confidence=0.85,
            ),
            CoachReviewAnnotation(
                analysis_id="bad",
                video="smash.mp4",
                priority_agreement=1,
                technical_correctness=1,
                actionability=2,
                clarity=2,
                drill_relevance=1,
                expected_priority_issue_codes=["INSUFFICIENT_ELBOW_EXTENSION"],
                analysis_confidence=0.3,
                comments="Invented biomechanics",
            ),
        ]
    )
    report = validate_coaching_reports(
        [
            ("ok", _correct_report(), _evidence(analysis_confidence=0.85)),
            ("bad", _contradictory_report(), _evidence(analysis_confidence=0.3)),
        ],
        coach_reviews=reviews,
    )
    assert report.analysis_count == 2
    assert report.hallucination_unsupported_rate == pytest.approx(0.5)
    assert report.evidence_faithfulness_rate is not None
    assert 0.0 < report.evidence_faithfulness_rate < 1.0
    assert report.mean_coach_ratings["priority_agreement"] == pytest.approx(3.0)
    assert report.mean_priority_agreement_rate == pytest.approx(0.5)

    by_conf = {g.key: g for g in report.by_analysis_confidence}
    assert by_conf["high_>=0.75"].sample_count == 1
    assert by_conf["low_<0.5"].sample_count == 1
    assert by_conf["high_>=0.75"].hallucination_unsupported_rate == pytest.approx(0.0)
    assert by_conf["low_<0.5"].hallucination_unsupported_rate == pytest.approx(1.0)

    out = tmp_path / "coaching_validation.json"
    export_coaching_validation_report(report, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (
        data["coaching_validation_report_version"]
        == COACHING_VALIDATION_REPORT_VERSION
    )
    assert data["analysis_count"] == 2
    assert len(data["analyses"]) == 2
    assert "evidence_faithfulness_rate" in data
    assert "hallucination_unsupported_rate" in data
    assert "mean_coach_ratings" in data
    assert "by_analysis_confidence" in data


def test_priority_agreement_uses_expected_codes_when_annotated() -> None:
    review = CoachReviewAnnotation(
        analysis_id="ok",
        expected_priority_issue_codes=[
            "INSUFFICIENT_ELBOW_EXTENSION",
            "LOW_KNEE_CONTRIBUTION",
        ],
    )
    record = CoachingValidator().validate_analysis(
        analysis_id="ok",
        report=_correct_report(),
        evidence=_evidence(),
        coach_review=review,
    )
    # Only 1 of 2 expected priorities present in the report.
    assert record.priority_agreement_rate == pytest.approx(0.5)


def test_load_example_coaching_annotations() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "pose_validation"
        / "example_coaching_annotations.json"
    )
    loaded = load_coaching_annotation_set(path)
    assert len(loaded.reviews) == 1
    review = loaded.reviews[0]
    assert review.priority_agreement == 4
    assert review.drill_relevance == 4
    assert review.expected_priority_issue_codes == [
        "INSUFFICIENT_ELBOW_EXTENSION"
    ]


def test_invalid_rating_rejected() -> None:
    with pytest.raises(ValueError, match="1–5"):
        CoachReviewAnnotation(analysis_id="x", clarity=6)
