"""C4 calibrated severity / confidence tests."""

from __future__ import annotations

import pytest

from app.processing.reference_profiles import METRIC_CONTACT_ELBOW
from app.processing.technique import TechniqueDecisionConfig, evaluate_technique
from app.schemas.reference import MetricReference, ReferenceProfile
from app.schemas.technique_calibration import (
    IssueStatus,
    SeverityCalibrationConfig,
    build_evaluation_confidence,
    calibrate_issue_status,
    combine_confidence,
    default_severity_calibration,
)
from tests.test_technique import _build_pipeline
from tests.test_technique_reference_distribution import _built_profile


def _elbow_only_profile(
    *,
    sample_count: int = 40,
    confidence: float = 0.85,
    provisional: bool = False,
    elbow_p10: float = 150.0,
    elbow_median: float = 162.0,
    elbow_std: float = 6.0,
) -> ReferenceProfile:
    base = _built_profile(
        elbow_p10=elbow_p10, elbow_median=elbow_median, elbow_std=elbow_std
    )
    elbow = base.metrics[METRIC_CONTACT_ELBOW]
    metrics = {
        METRIC_CONTACT_ELBOW: MetricReference(
            metric_id=elbow.metric_id,
            unit=elbow.unit,
            median=elbow.median,
            mean=elbow.mean,
            std=elbow.std,
            lower_percentile=elbow.lower_percentile,
            upper_percentile=elbow.upper_percentile,
            sample_count=sample_count,
            provenance=elbow.provenance,
            confidence=confidence,
            provisional=provisional,
            direction=elbow.direction,
            iqr=elbow.iqr,
        )
    }
    return ReferenceProfile(
        profile_id=base.profile_id,
        stroke_type=base.stroke_type,
        handedness=base.handedness,
        camera_view=base.camera_view,
        skill_level=base.skill_level,
        metrics=metrics,
        provisional=provisional,
        profile_version=base.profile_version,
        source=base.source,
        status="VALIDATED" if not provisional else "DRAFT",
        sample_count=sample_count,
    )


def test_combine_confidence_methods_are_configurable() -> None:
    cfg_min = SeverityCalibrationConfig(combination="min")
    cfg_w = SeverityCalibrationConfig(combination="weighted_mean")
    cfg_g = SeverityCalibrationConfig(combination="geometric_mean")
    kwargs = dict(
        pose=1.0, phase=0.8, measurement=0.6, video_quality=0.9, reference=0.5
    )
    assert combine_confidence(**kwargs, config=cfg_min) == pytest.approx(0.5)
    assert combine_confidence(**kwargs, config=cfg_w) > combine_confidence(
        **kwargs, config=cfg_min
    )
    assert 0.0 < combine_confidence(**kwargs, config=cfg_g) < 1.0


def test_normal_reference_range_no_issue() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 165.0
    metrics.phase_confidence = 0.95
    evaluation = evaluate_technique(
        metrics, profile=_elbow_only_profile(), quality_confidence=0.95
    )
    assert not any(i.code == "INSUFFICIENT_ELBOW_EXTENSION" for i in evaluation.issues)


def test_mild_deviation_minor() -> None:
    _, _, _, _, metrics = _build_pipeline()
    # Slightly below median / near P10 → MINOR region.
    metrics.contact_elbow_angle_deg = 152.0
    metrics.phase_confidence = 0.95
    calib = SeverityCalibrationConfig(
        major_percentile_max=1.0,
        moderate_percentile_max=5.0,
        minor_percentile_max=15.0,
        central_percentile_low=20.0,
        minor_abs_z=1.0,
        moderate_abs_z=2.0,
        major_abs_z=3.5,
    )
    evaluation = evaluate_technique(
        metrics,
        profile=_elbow_only_profile(elbow_p10=150.0, elbow_median=162.0, elbow_std=8.0),
        quality_confidence=0.95,
        decision_config=TechniqueDecisionConfig(severity_calibration=calib),
    )
    issue = next(i for i in evaluation.issues if i.code == "INSUFFICIENT_ELBOW_EXTENSION")
    assert issue.status == IssueStatus.MINOR.value
    assert issue.combined_confidence >= 0.5
    assert issue.severity_calibration_version == calib.version


def test_strong_deviation_moderate() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 140.0
    metrics.phase_confidence = 0.95
    calib = SeverityCalibrationConfig(
        major_abs_z=3.5,
        moderate_abs_z=1.5,
        minor_abs_z=0.8,
    )
    evaluation = evaluate_technique(
        metrics,
        profile=_elbow_only_profile(elbow_std=8.0),
        quality_confidence=0.95,
        decision_config=TechniqueDecisionConfig(severity_calibration=calib),
    )
    issue = next(i for i in evaluation.issues if i.code == "INSUFFICIENT_ELBOW_EXTENSION")
    assert issue.status in {IssueStatus.MODERATE.value, IssueStatus.MAJOR.value}


def test_extreme_deviation_major() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 110.0
    metrics.phase_confidence = 0.95
    evaluation = evaluate_technique(
        metrics, profile=_elbow_only_profile(), quality_confidence=0.95
    )
    issue = next(i for i in evaluation.issues if i.code == "INSUFFICIENT_ELBOW_EXTENSION")
    assert issue.status == IssueStatus.MAJOR.value
    assert issue.reference_percentile is not None
    assert issue.reference_percentile < 5.0


def test_high_deviation_low_measurement_confidence_insufficient() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 110.0
    metrics.phase_confidence = 0.15
    evaluation = evaluate_technique(
        metrics,
        profile=_elbow_only_profile(),
        quality_confidence=0.15,
        pose_confidence=0.2,
    )
    issue = next(i for i in evaluation.issues if i.code == "INSUFFICIENT_ELBOW_EXTENSION")
    assert issue.status == IssueStatus.INSUFFICIENT_EVIDENCE.value
    assert issue.combined_confidence < 0.5
    assert "confidence" in issue.status_reason.lower() or "confidence" in issue.description.lower()


def test_good_measurement_low_quality_reference_insufficient() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 110.0
    metrics.phase_confidence = 0.95
    evaluation = evaluate_technique(
        metrics,
        profile=_elbow_only_profile(confidence=0.1, provisional=True),
        quality_confidence=0.95,
    )
    issue = next(i for i in evaluation.issues if i.code == "INSUFFICIENT_ELBOW_EXTENSION")
    assert issue.status == IssueStatus.INSUFFICIENT_EVIDENCE.value


def test_insufficient_sample_count() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 110.0
    metrics.phase_confidence = 0.95
    evaluation = evaluate_technique(
        metrics,
        profile=_elbow_only_profile(sample_count=3, confidence=0.9),
        quality_confidence=0.95,
    )
    issue = next(i for i in evaluation.issues if i.code == "INSUFFICIENT_ELBOW_EXTENSION")
    assert issue.status == IssueStatus.INSUFFICIENT_EVIDENCE.value
    assert "sample_count" in issue.status_reason


def test_missing_metric_skipped() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 110.0
    metrics.knee_contribution_deg = 2.0
    metrics.phase_confidence = 0.95
    # Elbow-only profile → knee metric missing → no knee issue.
    evaluation = evaluate_technique(
        metrics, profile=_elbow_only_profile(), quality_confidence=0.95
    )
    assert not any(i.code == "LOW_KNEE_CONTRIBUTION" for i in evaluation.issues)
    assert any(i.code == "INSUFFICIENT_ELBOW_EXTENSION" for i in evaluation.issues)


def test_reproducible_with_same_config() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 130.0
    metrics.phase_confidence = 0.9
    calib = default_severity_calibration()
    decision = TechniqueDecisionConfig(severity_calibration=calib)
    profile = _elbow_only_profile()
    a = evaluate_technique(
        metrics, profile=profile, quality_confidence=0.9, decision_config=decision
    )
    b = evaluate_technique(
        metrics, profile=profile, quality_confidence=0.9, decision_config=decision
    )
    assert a.to_dict()["issues"] == b.to_dict()["issues"]
    assert a.severity_calibration_version == b.severity_calibration_version == calib.version


def test_calibrate_issue_status_central_is_no_issue() -> None:
    conf = build_evaluation_confidence(
        measurement_confidence=0.9,
        phase_confidence=0.9,
        video_quality_confidence=0.9,
        reference_confidence=0.8,
    )
    status, reason = calibrate_issue_status(
        direction="higher_is_better",
        percentile_position=45.0,
        robust_z=0.1,
        eval_confidence=conf,
        reference_sample_count=40,
        reference_provisional=False,
    )
    assert status == IssueStatus.NO_ISSUE
    assert "central" in reason.lower()


def test_rejects_literal_latest_profile_id() -> None:
    _, _, _, _, metrics = _build_pipeline()
    with pytest.raises(ValueError, match="latest"):
        evaluate_technique(metrics, profiles=[], profile_id="latest")
