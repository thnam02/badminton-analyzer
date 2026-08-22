"""Tests for V1 rule-based smash phase detection."""

from __future__ import annotations

from app.processing.motion import compute_motion_derivatives
from app.processing.phases import detect_smash_phases
from app.schemas.angles import AngleFrame, AngleSequence
from app.schemas.motion import MotionFrame, MotionSequence, PeakStats
from app.schemas.phases import SmashPhase
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence


def _kp(x: float, y: float, conf: float = 0.95) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=conf)


def _synthetic_smash(n: int = 40, contact: int = 28, dt: float = 0.05):
    """Build a smash-like wrist speed profile peaking at ``contact``."""
    pose_frames = []
    angle_frames = []
    motion_frames = []

    for i in range(n):
        t = i * dt
        # Relative profile: quiet → backswing rise → accel → peak → decay
        if i < 8:
            s_rel = 0.05
            elbow = 140.0
            omega = -20.0
        elif i < 18:
            s_rel = 0.05 + (i - 8) / 10 * 0.25
            elbow = 140.0 - (i - 8) * 2.0
            omega = -40.0
        elif i < contact:
            s_rel = 0.3 + (i - 18) / max(1, contact - 18) * 0.7
            elbow = 120.0 + (i - 18) * 3.0
            omega = 180.0
        elif i == contact:
            s_rel = 1.0
            elbow = 155.0
            omega = 220.0
        else:
            s_rel = max(0.05, 1.0 - (i - contact) * 0.12)
            elbow = 160.0
            omega = 40.0

        peak_speed = 2.0
        speed = s_rel * peak_speed
        # Wrist moves rightward with speed-integrated steps (for completeness).
        x = 0.3 + i * 0.01
        pose_frames.append(
            PoseFrame(
                frame_index=i,
                timestamp=t,
                keypoints={
                    "right_wrist": _kp(x, 0.4),
                    "right_elbow": _kp(x - 0.1, 0.35),
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
                right_knee=160.0,
                right_shoulder=90.0,
            )
        )
        motion_frames.append(
            MotionFrame(
                frame_index=i,
                timestamp=t,
                right_wrist_speed=None if i == 0 else speed,
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
    return pose, angles, motion, contact


def test_detects_estimated_contact_at_peak_wrist_speed() -> None:
    pose, angles, motion, contact = _synthetic_smash()
    phases = detect_smash_phases(pose, angles, motion)
    assert phases.estimated_contact_frame_index == contact
    assert phases.phase_at(contact) is SmashPhase.ESTIMATED_CONTACT
    assert "estimated" in phases.notes.lower() or "ESTIMATED_CONTACT" in phases.notes


def test_phase_order_around_contact() -> None:
    pose, angles, motion, contact = _synthetic_smash()
    phases = detect_smash_phases(pose, angles, motion)

    names = [seg.phase for seg in phases.segments]
    assert SmashPhase.ESTIMATED_CONTACT in names
    assert SmashPhase.FOLLOW_THROUGH in names
    # Before contact should include accel and usually prep/backswing.
    assert SmashPhase.ACCELERATION in names

    # Contact segment is a single frame.
    contact_seg = next(
        s for s in phases.segments if s.phase is SmashPhase.ESTIMATED_CONTACT
    )
    assert contact_seg.start_frame_index == contact_seg.end_frame_index == contact

    # A frame before contact is not FOLLOW_THROUGH.
    assert phases.phase_at(contact - 1) in {
        SmashPhase.ACCELERATION,
        SmashPhase.BACKSWING,
        SmashPhase.PREPARATION,
    }
    # After contact.
    assert phases.phase_at(min(contact + 2, 39)) is SmashPhase.FOLLOW_THROUGH


def test_segments_have_confidence_and_time_bounds() -> None:
    pose, angles, motion, _ = _synthetic_smash()
    phases = detect_smash_phases(pose, angles, motion)
    assert phases.confidence > 0
    for seg in phases.segments:
        assert 0.0 <= seg.confidence <= 1.0
        assert seg.start_frame_index <= seg.end_frame_index
        assert seg.start_timestamp <= seg.end_timestamp


def test_relative_not_absolute_still_finds_contact_with_scaled_speeds() -> None:
    # Same shape, much smaller absolute speeds — relative rules should still work.
    pose, angles, motion, contact = _synthetic_smash()
    for fr in motion.frames:
        if fr.right_wrist_speed is not None:
            fr.right_wrist_speed *= 0.05
    motion.peaks["right_wrist_speed"] = PeakStats(
        value=0.1,
        frame_index=contact,
        timestamp=contact * 0.05,
    )
    phases = detect_smash_phases(pose, angles, motion)
    assert phases.estimated_contact_frame_index == contact
    assert phases.phase_at(contact) is SmashPhase.ESTIMATED_CONTACT


def test_works_with_motion_derivatives_pipeline() -> None:
    # Sanity: phases after real motion derivative computation on simple motion.
    pose_frames = []
    angle_frames = []
    for i in range(12):
        t = i * 0.1
        x = 0.2 + i * 0.05
        pose_frames.append(
            PoseFrame(
                frame_index=i,
                timestamp=t,
                keypoints={
                    "right_wrist": _kp(x, 0.5),
                    "right_elbow": _kp(x - 0.1, 0.4),
                    "right_shoulder": _kp(0.2, 0.3),
                    "right_hip": _kp(0.25, 0.6),
                    "right_knee": _kp(0.25, 0.75),
                    "right_ankle": _kp(0.3, 0.9),
                },
            )
        )
        angle_frames.append(
            AngleFrame(
                frame_index=i,
                timestamp=t,
                right_elbow=100 + i * 5,
                right_knee=150,
                right_shoulder=90,
            )
        )
    pose = PoseSequence(video="clip.mp4", frames=pose_frames)
    angles = AngleSequence(video="clip.mp4", frames=angle_frames)
    motion = compute_motion_derivatives(pose, angles, confidence_threshold=0.5)
    phases = detect_smash_phases(pose, angles, motion)
    assert phases.estimated_contact_frame_index is not None
    assert phases.phase_at(phases.estimated_contact_frame_index) is SmashPhase.ESTIMATED_CONTACT
