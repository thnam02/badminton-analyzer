"""Tests for phase-specific stroke feature extraction."""

from __future__ import annotations

import pytest

from app.processing.phases import detect_smash_phases
from app.processing.stroke_metrics import extract_stroke_metrics
from app.schemas.angles import AngleSequence
from app.schemas.motion import MotionSequence
from app.schemas.phases import PhaseSequence, SmashPhase
from tests.test_phases import _synthetic_smash


def test_contact_metrics_use_estimated_contact_anchor() -> None:
    pose, angles, motion, contact = _synthetic_smash()
    phases = detect_smash_phases(pose, angles, motion)
    metrics = extract_stroke_metrics(phases, angles, motion)

    assert metrics.estimated_contact.frame_index == contact
    assert metrics.estimated_contact.right_elbow_angle == pytest.approx(155.0)
    assert metrics.estimated_contact.right_knee_angle == pytest.approx(160.0)
    assert metrics.estimated_contact.right_wrist_speed == pytest.approx(2.0)


def test_acceleration_peaks_are_windowed_not_global() -> None:
    pose, angles, motion, contact = _synthetic_smash()
    phases = detect_smash_phases(pose, angles, motion)
    metrics = extract_stroke_metrics(phases, angles, motion)

    accel = next(s for s in phases.segments if s.phase is SmashPhase.ACCELERATION)
    window_speeds = [
        f.right_wrist_speed
        for f in motion.frames
        if accel.start_frame_index <= f.frame_index <= accel.end_frame_index
        and f.right_wrist_speed is not None
    ]
    window_omega = [
        abs(f.right_elbow_angular_velocity)
        for f in motion.frames
        if accel.start_frame_index <= f.frame_index <= accel.end_frame_index
        and f.right_elbow_angular_velocity is not None
    ]

    assert metrics.acceleration.peak_wrist_speed.value == pytest.approx(max(window_speeds))
    assert metrics.acceleration.peak_elbow_angular_velocity.value == pytest.approx(
        max(window_omega)
    )
    # Global wrist-speed peak is at contact, so accel peak must be strictly lower.
    assert metrics.acceleration.peak_wrist_speed.value < 2.0
    assert metrics.acceleration.peak_wrist_speed.frame_index != contact
    assert metrics.acceleration.peak_elbow_angular_velocity.value == pytest.approx(180.0)


def test_preparation_and_follow_through_have_basic_metrics() -> None:
    pose, angles, motion, _ = _synthetic_smash()
    phases = detect_smash_phases(pose, angles, motion)
    metrics = extract_stroke_metrics(phases, angles, motion)

    assert SmashPhase.PREPARATION in {s.phase for s in phases.segments}
    assert SmashPhase.FOLLOW_THROUGH in {s.phase for s in phases.segments}

    assert metrics.preparation.window.duration is not None
    assert metrics.preparation.window.duration >= 0.0
    assert metrics.preparation.mean_wrist_speed is not None
    assert metrics.preparation.mean_elbow_angle is not None
    assert metrics.preparation.mean_knee_angle is not None

    assert metrics.follow_through.window.duration is not None
    assert metrics.follow_through.window.duration >= 0.0
    assert metrics.follow_through.mean_wrist_speed is not None
    assert metrics.follow_through.wrist_speed_at_end is not None
    assert metrics.follow_through.elbow_angle_at_end is not None
    assert metrics.follow_through.wrist_speed_at_end < metrics.estimated_contact.right_wrist_speed


def test_backswing_min_elbow_is_in_backswing_window() -> None:
    pose, angles, motion, _ = _synthetic_smash()
    phases = detect_smash_phases(pose, angles, motion)
    metrics = extract_stroke_metrics(phases, angles, motion)

    back = next(s for s in phases.segments if s.phase is SmashPhase.BACKSWING)
    idx = metrics.backswing.min_elbow_angle.frame_index
    assert idx is not None
    assert back.start_frame_index <= idx <= back.end_frame_index
    assert metrics.backswing.min_elbow_angle.value is not None


def test_missing_phases_yield_null_metrics() -> None:
    metrics = extract_stroke_metrics(
        PhaseSequence(video="empty.mp4"),
        AngleSequence(video="empty.mp4"),
        MotionSequence(video="empty.mp4"),
    )
    assert metrics.estimated_contact.right_elbow_angle is None
    assert metrics.estimated_contact.right_wrist_speed is None
    assert metrics.acceleration.peak_wrist_speed.value is None
    assert metrics.acceleration.peak_elbow_angular_velocity.value is None
    assert metrics.preparation.mean_wrist_speed is None
    assert metrics.follow_through.wrist_speed_at_end is None
    assert "scoring" not in metrics.to_dict()
    assert "feedback" not in metrics.to_dict()
