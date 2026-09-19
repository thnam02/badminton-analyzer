"""Tests for stroke metrics and profile-based technique evaluation."""

from __future__ import annotations

from app.processing.phases import detect_smash_phases
from app.processing.reference_profiles import (
    METRIC_ACCEL_FRACTION,
    METRIC_CONTACT_ELBOW,
    METRIC_CONTACT_WRIST_Y,
    METRIC_ELBOW_PEAK_TIMING,
    METRIC_FOLLOW_THROUGH_FRAMES,
    METRIC_FOLLOW_THROUGH_RETENTION,
    METRIC_KNEE_CONTRIBUTION,
    build_provisional_smash_right_side,
)
from app.processing.stroke_metrics import compute_stroke_metrics
from app.processing.technique import evaluate_technique
from app.schemas.angles import AngleFrame, AngleSequence
from app.schemas.motion import MotionFrame, MotionSequence, PeakStats
from app.schemas.phases import SmashPhase
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence
from app.schemas.reference import MetricReference, ReferenceProfile
from app.schemas.technique import IssueSeverity


def _kp(x: float, y: float, conf: float = 0.95) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=conf)


def _build_pipeline(n: int = 40, contact: int = 28, dt: float = 0.05):
    pose_frames = []
    angle_frames = []
    motion_frames = []
    for i in range(n):
        t = i * dt
        if i < 8:
            s_rel, elbow, omega, knee = 0.05, 140.0, -20.0, 145.0
            wrist_y = 0.35
        elif i < 18:
            s_rel = 0.05 + (i - 8) / 10 * 0.25
            elbow = 140.0 - (i - 8) * 2.0
            omega, knee, wrist_y = -40.0, 140.0, 0.38
        elif i < contact:
            s_rel = 0.3 + (i - 18) / max(1, contact - 18) * 0.7
            elbow = 120.0 + (i - 18) * 3.0
            omega, knee, wrist_y = 180.0, 155.0, 0.42
        elif i == contact:
            s_rel, elbow, omega, knee, wrist_y = 1.0, 155.0, 220.0, 158.0, 0.44
        else:
            s_rel = max(0.05, 1.0 - (i - contact) * 0.12)
            elbow, omega, knee, wrist_y = 160.0, 40.0, 160.0, 0.46

        x = 0.3 + i * 0.01
        pose_frames.append(
            PoseFrame(
                frame_index=i,
                timestamp=t,
                keypoints={
                    "right_wrist": _kp(x, wrist_y),
                    "right_elbow": _kp(x - 0.1, wrist_y - 0.05),
                    "right_shoulder": _kp(x - 0.2, 0.3),
                    "right_hip": _kp(x - 0.15, 0.55),
                    "right_knee": _kp(x - 0.15, 0.7),
                    "right_ankle": _kp(x - 0.1, 0.85),
                },
            )
        )
        angle_frames.append(
            AngleFrame(
                frame_index=i,
                timestamp=t,
                right_elbow=elbow,
                right_knee=knee,
                right_shoulder=90.0,
            )
        )
        motion_frames.append(
            MotionFrame(
                frame_index=i,
                timestamp=t,
                right_wrist_speed=None if i == 0 else s_rel * 2.0,
                right_elbow_angular_velocity=None if i == 0 else omega,
                right_knee_angular_velocity=0.0 if i else None,
            )
        )

    pose = PoseSequence(video="smash.mp4", frames=pose_frames)
    angles = AngleSequence(video="smash.mp4", frames=angle_frames)
    motion = MotionSequence(
        video="smash.mp4",
        frames=motion_frames,
        peaks={
            "right_wrist_speed": PeakStats(
                value=2.0,
                frame_index=contact,
                timestamp=contact * dt,
            )
        },
    )
    phases = detect_smash_phases(pose, angles, motion)
    metrics = compute_stroke_metrics(pose, angles, motion, phases)
    return pose, angles, motion, phases, metrics


def _profile_with(**metric_overrides: MetricReference) -> ReferenceProfile:
    base = build_provisional_smash_right_side()
    metrics = dict(base.metrics)
    metrics.update(metric_overrides)
    return ReferenceProfile(
        profile_id=base.profile_id,
        stroke_type=base.stroke_type,
        handedness=base.handedness,
        camera_view=base.camera_view,
        skill_level=base.skill_level,
        metrics=metrics,
        provisional=True,
        profile_version=base.profile_version,
        source=base.source,
        notes=base.notes,
    )


def test_stroke_metrics_populated_at_contact() -> None:
    _, _, _, phases, metrics = _build_pipeline()
    assert metrics.estimated_contact_frame_index == phases.estimated_contact_frame_index
    assert metrics.contact_elbow_angle_deg is not None
    assert metrics.peak_wrist_speed is not None
    assert metrics.knee_contribution_deg is not None
    payload = metrics.to_dict()
    assert payload["video"] == "smash.mp4"
    assert "contact_elbow_angle_deg" in payload


def test_good_technique_produces_few_or_no_issues() -> None:
    _, _, _, _, metrics = _build_pipeline()
    evaluation = evaluate_technique(metrics, profile=build_provisional_smash_right_side())
    assert evaluation.issue_count <= 3
    assert evaluation.reference_profile_id == "smash_right_side_provisional_v1"


def test_insufficient_elbow_extension_detected() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 120.0
    profile = _profile_with(
        **{
            METRIC_CONTACT_ELBOW: MetricReference(
                metric_id=METRIC_CONTACT_ELBOW,
                unit="deg",
                median=162.0,
                lower_percentile=150.0,
                upper_percentile=180.0,
                sample_count=24,
                provenance="test",
                confidence=0.5,
                provisional=True,
                direction="higher_is_better",
            )
        }
    )
    evaluation = evaluate_technique(metrics, profile=profile)
    codes = {i.code for i in evaluation.issues}
    assert "INSUFFICIENT_ELBOW_EXTENSION" in codes
    issue = next(i for i in evaluation.issues if i.code == "INSUFFICIENT_ELBOW_EXTENSION")
    assert issue.phase is SmashPhase.ESTIMATED_CONTACT
    assert issue.measured_value == 120.0
    assert issue.reference_range.min == 150.0
    assert issue.unit == "deg"
    assert issue.severity in IssueSeverity
    assert issue.reference_profile_id == profile.profile_id
    assert issue.reference_evidence is not None
    assert issue.reference_evidence.metric_id == METRIC_CONTACT_ELBOW
    assert issue.reference_evidence.provisional is True
    assert issue.reference_evidence.lower_percentile == 150.0


def test_low_knee_contribution_detected() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.knee_contribution_deg = 3.0
    profile = _profile_with(
        **{
            METRIC_KNEE_CONTRIBUTION: MetricReference(
                metric_id=METRIC_KNEE_CONTRIBUTION,
                unit="deg",
                median=18.0,
                lower_percentile=12.0,
                upper_percentile=40.0,
                sample_count=24,
                provenance="test",
                confidence=0.5,
                provisional=True,
                direction="higher_is_better",
            )
        }
    )
    evaluation = evaluate_technique(metrics, profile=profile)
    assert any(i.code == "LOW_KNEE_CONTRIBUTION" for i in evaluation.issues)


def test_poor_acceleration_timing_detected() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.peak_elbow_omega_offset_frames = 10
    profile = _profile_with(
        **{
            METRIC_ELBOW_PEAK_TIMING: MetricReference(
                metric_id=METRIC_ELBOW_PEAK_TIMING,
                unit="frames",
                median=-2.0,
                lower_percentile=-8.0,
                upper_percentile=2.0,
                sample_count=24,
                provenance="test",
                confidence=0.5,
                provisional=True,
                direction="in_range",
            )
        }
    )
    evaluation = evaluate_technique(metrics, profile=profile)
    assert any(i.code == "POOR_ARM_ACCELERATION_TIMING" for i in evaluation.issues)


def test_low_contact_posture_detected() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_wrist_y_normalized = 0.75
    profile = _profile_with(
        **{
            METRIC_CONTACT_WRIST_Y: MetricReference(
                metric_id=METRIC_CONTACT_WRIST_Y,
                unit="normalized_y",
                median=0.42,
                lower_percentile=0.15,
                upper_percentile=0.58,
                sample_count=24,
                provenance="test",
                confidence=0.5,
                provisional=True,
                direction="lower_is_better",
            )
        }
    )
    evaluation = evaluate_technique(metrics, profile=profile)
    issue = next(i for i in evaluation.issues if i.code == "LOW_CONTACT_POSTURE")
    assert issue.measured_value == 0.75
    assert issue.reference_range.max == 0.58


def test_weak_follow_through_detected() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.follow_through_speed_ratio = 0.1
    metrics.follow_through_frame_count = 1
    profile = _profile_with(
        **{
            METRIC_FOLLOW_THROUGH_RETENTION: MetricReference(
                metric_id=METRIC_FOLLOW_THROUGH_RETENTION,
                unit="speed_ratio",
                median=0.45,
                lower_percentile=0.30,
                upper_percentile=1.0,
                sample_count=24,
                provenance="test",
                confidence=0.5,
                provisional=True,
                direction="higher_is_better",
            ),
            METRIC_FOLLOW_THROUGH_FRAMES: MetricReference(
                metric_id=METRIC_FOLLOW_THROUGH_FRAMES,
                unit="frames",
                median=6.0,
                lower_percentile=2.0,
                upper_percentile=20.0,
                sample_count=24,
                provenance="test",
                confidence=0.5,
                provisional=True,
                direction="higher_is_better",
            ),
            METRIC_ACCEL_FRACTION: build_provisional_smash_right_side().metrics[
                METRIC_ACCEL_FRACTION
            ],
        }
    )
    evaluation = evaluate_technique(metrics, profile=profile)
    assert any(i.code == "WEAK_FOLLOW_THROUGH" for i in evaluation.issues)


def test_technique_issue_has_required_fields() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 100.0
    metrics.phase_confidence = 0.9
    evaluation = evaluate_technique(
        metrics,
        profile=build_provisional_smash_right_side(),
        quality_confidence=0.9,
    )
    assert evaluation.issues
    issue = evaluation.issues[0]
    assert issue.code
    assert issue.phase
    assert issue.severity
    assert 0.0 <= issue.confidence <= 1.0
    assert issue.reference_range.min is not None or issue.reference_range.max is not None
    assert issue.unit
    assert issue.reference_profile_id
    assert issue.metric_name
    assert issue.rule_version
    assert issue.decision_mode == "reference_distribution"
    assert issue.measurement_confidence > 0
    assert issue.reference_evidence is not None
    assert "provisional" in issue.to_dict()["reference_evidence"]
