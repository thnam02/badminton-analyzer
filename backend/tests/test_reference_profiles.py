"""Tests for reference profile selection and percentile-based detection."""

from __future__ import annotations

import pytest

from app.processing.reference_profiles import (
    METRIC_CONTACT_ELBOW,
    METRIC_PREP_KNEE,
    build_provisional_smash_any,
    build_provisional_smash_left_side,
    build_provisional_smash_right_side,
    default_reference_profiles,
    select_reference_profile,
)
from app.processing.technique import evaluate_technique
from app.schemas.reference import MetricReference, ReferenceProfile
from app.schemas.stroke_metrics import StrokeMetrics
from tests.test_technique import _build_pipeline


def test_select_exact_stroke_hand_view() -> None:
    profile = select_reference_profile(
        stroke_type="SMASH",
        handedness="RIGHT",
        camera_view="SIDE",
    )
    assert profile.profile_id == "smash_right_side_provisional_v1"
    assert profile.handedness == "RIGHT"
    assert profile.camera_view == "SIDE"
    assert profile.provisional is True


def test_select_left_handed_side_view() -> None:
    profile = select_reference_profile(
        stroke_type="SMASH",
        handedness="LEFT",
        camera_view="SIDE",
    )
    assert profile.profile_id == "smash_left_side_provisional_v1"


def test_select_by_explicit_profile_id() -> None:
    profile = select_reference_profile(profile_id="smash_any_provisional_v1")
    assert profile.profile_id == "smash_any_provisional_v1"
    assert profile.handedness is None


def test_unknown_profile_id_raises() -> None:
    with pytest.raises(KeyError):
        select_reference_profile(profile_id="does_not_exist")


def test_fallback_when_handedness_unknown() -> None:
    profile = select_reference_profile(
        stroke_type="SMASH",
        handedness=None,
        camera_view=None,
    )
    assert profile.stroke_type == "SMASH"
    # Prefer wildcard any-profile when hand/view unknown.
    assert profile.profile_id == "smash_any_provisional_v1"


def test_provisional_profiles_mark_sample_provenance() -> None:
    for profile in default_reference_profiles():
        assert profile.provisional is True
        assert "not scientifically validated" in profile.notes.lower()
        elbow = profile.get_metric(METRIC_CONTACT_ELBOW)
        assert elbow is not None
        assert elbow.provisional is True
        assert "provisional" in elbow.provenance


def test_percentile_higher_is_better_flags_below_p10() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 140.0  # below provisional p10=150
    profile = build_provisional_smash_right_side()
    evaluation = evaluate_technique(metrics, profile=profile)
    issue = next(i for i in evaluation.issues if i.code == "INSUFFICIENT_ELBOW_EXTENSION")
    assert issue.reference_evidence is not None
    assert issue.reference_evidence.lower_percentile == pytest.approx(150.0)
    assert issue.measured_value < issue.reference_evidence.lower_percentile


def test_percentile_in_range_flags_outside_band() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.peak_elbow_omega_offset_frames = 12  # above p90=2
    profile = build_provisional_smash_right_side()
    evaluation = evaluate_technique(metrics, profile=profile)
    assert any(i.code == "POOR_ARM_ACCELERATION_TIMING" for i in evaluation.issues)


def test_percentile_in_range_prep_knee() -> None:
    metrics = StrokeMetrics(
        video="unit.mp4",
        phase_confidence=0.8,
        estimated_contact_frame_index=10,
        preparation_knee_angle_deg=100.0,  # below p10=120
    )
    base = build_provisional_smash_right_side()
    evaluation = evaluate_technique(metrics, profile=base)
    issue = next(i for i in evaluation.issues if i.code == "PREPARATION_KNEE_OUT_OF_RANGE")
    assert issue.reference_evidence is not None
    assert issue.reference_evidence.metric_id == METRIC_PREP_KNEE
    assert issue.reference_range.min == pytest.approx(120.0)


def test_value_inside_percentile_band_is_not_flagged() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 165.0
    metrics.knee_contribution_deg = 20.0
    metrics.peak_elbow_omega_offset_frames = -2
    metrics.follow_through_speed_ratio = 0.5
    metrics.follow_through_frame_count = 8
    metrics.contact_wrist_y_normalized = 0.4
    metrics.preparation_knee_angle_deg = 140.0
    metrics.acceleration_phase_fraction = 0.3
    evaluation = evaluate_technique(metrics, profile=build_provisional_smash_right_side())
    assert evaluation.issue_count == 0


def test_custom_catalog_selection() -> None:
    custom = ReferenceProfile(
        profile_id="custom_front",
        stroke_type="SMASH",
        handedness="RIGHT",
        camera_view="FRONT",
        metrics={
            METRIC_CONTACT_ELBOW: MetricReference(
                metric_id=METRIC_CONTACT_ELBOW,
                unit="deg",
                median=160.0,
                lower_percentile=145.0,
                upper_percentile=180.0,
                direction="higher_is_better",
                provisional=True,
                provenance="unit_test",
                sample_count=3,
                confidence=0.2,
            )
        },
        provisional=True,
    )
    selected = select_reference_profile(
        stroke_type="SMASH",
        handedness="RIGHT",
        camera_view="FRONT",
        profiles=[custom, build_provisional_smash_any(), build_provisional_smash_left_side()],
    )
    assert selected.profile_id == "custom_front"
