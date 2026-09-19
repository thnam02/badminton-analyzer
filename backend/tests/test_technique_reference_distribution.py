"""Tests for ReferenceProfileSelector and distribution-aware technique evaluation."""

from __future__ import annotations

import pytest

from app.processing.reference_profile_selector import (
    MATCH_EXACT,
    MATCH_NONE,
    MATCH_SKILL_FALLBACK,
    MATCH_STROKE,
    ReferenceProfileSelector,
)
from app.processing.reference_profiles import (
    METRIC_CONTACT_ELBOW,
    METRIC_ELBOW_PEAK_TIMING,
    build_provisional_smash_any,
    build_provisional_smash_right_side,
)
from app.processing.technique import TechniqueDecisionConfig, evaluate_technique
from app.schemas.reference import MetricReference, ReferenceProfile
from app.schemas.stroke_metrics import StrokeMetrics
from app.schemas.technique import (
    TECHNIQUE_RULE_VERSION_FALLBACK,
    TECHNIQUE_RULE_VERSION_REFERENCE,
)
from tests.test_technique import _build_pipeline, _profile_with


def _built_profile(
    *,
    profile_id: str = "smash_right_side_45_intermediate_built_v1_0_0",
    skill_level: str = "intermediate",
    camera_view: str = "side_45",
    handedness: str = "RIGHT",
    elbow_p10: float = 150.0,
    elbow_median: float = 162.0,
    elbow_std: float = 6.0,
    version: str = "1.0.0",
) -> ReferenceProfile:
    return ReferenceProfile(
        profile_id=profile_id,
        stroke_type="SMASH",
        handedness=handedness,
        camera_view=camera_view,
        skill_level=skill_level,
        provisional=False,
        profile_version=version,
        source="built_reference",
        metrics={
            METRIC_CONTACT_ELBOW: MetricReference(
                metric_id=METRIC_CONTACT_ELBOW,
                unit="deg",
                median=elbow_median,
                mean=elbow_median,
                std=elbow_std,
                lower_percentile=elbow_p10,
                upper_percentile=180.0,
                lower_percentile_rank=10.0,
                upper_percentile_rank=90.0,
                sample_count=40,
                provenance="built_from_reference_dataset",
                confidence=0.8,
                provisional=False,
                direction="higher_is_better",
                iqr=10.0,
            ),
            METRIC_ELBOW_PEAK_TIMING: MetricReference(
                metric_id=METRIC_ELBOW_PEAK_TIMING,
                unit="frames",
                median=-2.0,
                mean=-2.0,
                std=1.5,
                lower_percentile=-8.0,
                upper_percentile=2.0,
                sample_count=40,
                provenance="built_from_reference_dataset",
                confidence=0.8,
                provisional=False,
                direction="in_range",
            ),
        },
        notes="test built profile",
    )


def test_selector_exact_match_with_skill() -> None:
    catalog = [
        _built_profile(skill_level="beginner", profile_id="beginner"),
        _built_profile(skill_level="intermediate", profile_id="intermediate"),
        build_provisional_smash_any(),
    ]
    selection = ReferenceProfileSelector(catalog).select(
        stroke_type="SMASH",
        handedness="RIGHT",
        camera_view="side_45",
        skill_level="intermediate",
    )
    assert selection.match_level == MATCH_EXACT
    assert selection.profile is not None
    assert selection.profile.profile_id == "intermediate"


def test_selector_skill_fallback_when_skill_missing_on_profile() -> None:
    catalog = [
        ReferenceProfile(
            profile_id="hand_view_only",
            stroke_type="SMASH",
            handedness="RIGHT",
            camera_view="side_45",
            skill_level=None,
            metrics=dict(_built_profile().metrics),
            provisional=False,
            profile_version="1.0.0",
            source="built_reference",
        )
    ]
    selection = ReferenceProfileSelector(catalog).select(
        stroke_type="FOREHAND_SMASH",
        handedness="RIGHT",
        camera_view="SIDE",
        skill_level="advanced",
    )
    assert selection.has_valid_profile
    assert selection.match_level == MATCH_SKILL_FALLBACK
    assert selection.profile is not None
    assert selection.profile.profile_id == "hand_view_only"


def test_selector_returns_none_when_catalog_empty() -> None:
    selection = ReferenceProfileSelector([]).select(
        stroke_type="SMASH", handedness="RIGHT", camera_view="side_45"
    )
    assert selection.match_level == MATCH_NONE
    assert selection.profile is None


def test_percentile_issue_detection_with_built_profile() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 140.0  # below P10=150
    metrics.phase_confidence = 0.9
    profile = _built_profile()
    evaluation = evaluate_technique(
        metrics,
        profile=profile,
        quality_confidence=0.9,
    )
    issue = next(i for i in evaluation.issues if i.code == "INSUFFICIENT_ELBOW_EXTENSION")
    assert evaluation.decision_mode == "reference_distribution"
    assert evaluation.rule_version == TECHNIQUE_RULE_VERSION_REFERENCE
    assert issue.metric_name == METRIC_CONTACT_ELBOW
    assert issue.reference_median == pytest.approx(162.0)
    assert issue.deviation == pytest.approx(140.0 - 162.0)
    assert issue.percentile_position is not None
    assert issue.percentile_position < 10.0 or issue.measured_value < 150.0
    assert issue.measurement_confidence >= 0.85
    assert issue.rule_version == TECHNIQUE_RULE_VERSION_REFERENCE
    assert issue.reference_evidence is not None
    assert issue.reference_evidence.robust_z is not None
    assert issue.uncertain is False


def test_normal_reference_produces_no_elbow_issue() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 165.0
    metrics.knee_contribution_deg = 20.0
    metrics.peak_elbow_omega_offset_frames = -2
    metrics.follow_through_speed_ratio = 0.5
    metrics.follow_through_frame_count = 8
    metrics.contact_wrist_y_normalized = 0.4
    metrics.preparation_knee_angle_deg = 140.0
    metrics.acceleration_phase_fraction = 0.3
    metrics.phase_confidence = 0.9
    evaluation = evaluate_technique(
        metrics,
        profile=_built_profile(),
        quality_confidence=0.9,
        # Use full provisional metrics via merge for other checks:
        # built profile only has elbow+timing — other metrics absent → no issue.
    )
    assert not any(i.code == "INSUFFICIENT_ELBOW_EXTENSION" for i in evaluation.issues)
    assert not any(i.code == "POOR_ARM_ACCELERATION_TIMING" for i in evaluation.issues)


def test_timing_issue_uses_reference_distribution() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.peak_elbow_omega_offset_frames = 10  # above P90=2
    metrics.phase_confidence = 0.9
    evaluation = evaluate_technique(
        metrics, profile=_built_profile(), quality_confidence=0.9
    )
    issue = next(i for i in evaluation.issues if i.code == "POOR_ARM_ACCELERATION_TIMING")
    assert issue.metric_name == METRIC_ELBOW_PEAK_TIMING
    assert issue.decision_mode == "reference_distribution"
    assert issue.reference_range.max == pytest.approx(2.0)


def test_low_confidence_suppressed() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 120.0
    metrics.phase_confidence = 0.1
    evaluation = evaluate_technique(
        metrics,
        profile=_built_profile(),
        quality_confidence=0.1,
        decision_config=TechniqueDecisionConfig(suppress_below_confidence=0.25),
    )
    assert not any(i.code == "INSUFFICIENT_ELBOW_EXTENSION" for i in evaluation.issues)


def test_low_confidence_uncertain_not_strong() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 120.0
    metrics.phase_confidence = 0.4
    evaluation = evaluate_technique(
        metrics,
        profile=_built_profile(),
        quality_confidence=0.4,
        decision_config=TechniqueDecisionConfig(
            suppress_below_confidence=0.25,
            uncertain_below_confidence=0.45,
        ),
    )
    issue = next(i for i in evaluation.issues if i.code == "INSUFFICIENT_ELBOW_EXTENSION")
    assert issue.uncertain is True
    assert issue.severity.value == "LOW"


def test_provisional_fallback_when_no_profile() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 120.0
    metrics.phase_confidence = 0.9
    evaluation = evaluate_technique(
        metrics,
        profiles=[],  # empty catalog → none → hardcoded fallback
        quality_confidence=0.9,
    )
    assert evaluation.decision_mode == "provisional_fallback"
    assert evaluation.rule_version == TECHNIQUE_RULE_VERSION_FALLBACK
    assert evaluation.profile_match_level == MATCH_NONE
    issue = next(i for i in evaluation.issues if i.code == "INSUFFICIENT_ELBOW_EXTENSION")
    assert issue.decision_mode == "provisional_fallback"
    assert issue.rule_version == TECHNIQUE_RULE_VERSION_FALLBACK
    assert "provisional fallback" in issue.description.lower()
    assert issue.reference_evidence is None


def test_does_not_mix_fallback_thresholds_with_profile_decisions() -> None:
    """Profile with high P10 must not use settings threshold (150) silently."""
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 155.0  # above settings 150, below profile P10=160
    metrics.phase_confidence = 0.9
    profile = _built_profile(elbow_p10=160.0, elbow_median=170.0)
    evaluation = evaluate_technique(
        metrics, profile=profile, quality_confidence=0.9
    )
    assert any(i.code == "INSUFFICIENT_ELBOW_EXTENSION" for i in evaluation.issues)
    issue = next(i for i in evaluation.issues if i.code == "INSUFFICIENT_ELBOW_EXTENSION")
    assert issue.reference_range.min == pytest.approx(160.0)
    assert issue.decision_mode == "reference_distribution"


def test_reproducible_across_profile_versions() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 140.0
    metrics.phase_confidence = 0.9
    v1 = _built_profile(version="1.0.0", profile_id="p_v1")
    v2 = _built_profile(version="1.0.0", profile_id="p_v1")  # identical content
    a = evaluate_technique(metrics, profile=v1, quality_confidence=0.9)
    b = evaluate_technique(metrics, profile=v2, quality_confidence=0.9)
    assert a.to_dict()["issues"] == b.to_dict()["issues"]
    assert a.rule_version == b.rule_version == TECHNIQUE_RULE_VERSION_REFERENCE

    v2_changed = _built_profile(
        version="2.0.0",
        elbow_p10=130.0,
        elbow_median=145.0,
        elbow_std=4.0,
        profile_id="p_v2",
    )
    c = evaluate_technique(
        metrics,
        profile=v2_changed,
        quality_confidence=0.9,
        decision_config=TechniqueDecisionConfig(use_robust_z=True),
    )
    # 140 is above new P10=130 and within ~1.25σ of median → no elbow issue.
    assert not any(i.code == "INSUFFICIENT_ELBOW_EXTENSION" for i in c.issues)


def test_selector_stroke_wildcard() -> None:
    catalog = [build_provisional_smash_any()]
    selection = ReferenceProfileSelector(catalog).select(
        stroke_type="SMASH",
        handedness="RIGHT",
        camera_view="rear_45",
    )
    # RIGHT vs None hand is compatible (score 0), rear vs None view compatible.
    assert selection.has_valid_profile
    assert selection.profile is not None
    assert selection.profile.profile_id == "smash_any_provisional_v1"
    assert selection.match_level in {MATCH_STROKE, MATCH_EXACT, "hand_fallback", "view_fallback"}
