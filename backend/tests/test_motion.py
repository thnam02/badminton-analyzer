"""Unit tests for motion derivatives (speed / angular velocity / peaks)."""

from __future__ import annotations

import pytest

from app.processing.motion import compute_motion_derivatives
from app.schemas.angles import AngleFrame, AngleSequence
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence


def _kp(x: float, y: float, confidence: float = 0.95) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=confidence)


def _pose_frame(
    index: int,
    t: float,
    *,
    wrist: tuple[float, float] | None = None,
    wrist_conf: float = 0.95,
) -> PoseFrame:
    keypoints = {}
    if wrist is not None:
        keypoints["right_wrist"] = _kp(wrist[0], wrist[1], wrist_conf)
    return PoseFrame(frame_index=index, timestamp=t, keypoints=keypoints)


def _angle_frame(
    index: int,
    t: float,
    *,
    elbow: float | None = None,
    knee: float | None = None,
) -> AngleFrame:
    return AngleFrame(
        frame_index=index,
        timestamp=t,
        right_elbow=elbow,
        right_knee=knee,
        right_shoulder=None,
    )


def test_right_wrist_linear_speed_uses_timestamps() -> None:
    # Move 0.3 in x over 0.1s → speed 3.0 norm-units/s. Uneven dt on next step.
    pose = PoseSequence(
        video="clip.mp4",
        frames=[
            _pose_frame(0, 0.00, wrist=(0.10, 0.50)),
            _pose_frame(1, 0.10, wrist=(0.40, 0.50)),
            _pose_frame(2, 0.30, wrist=(0.40, 0.80)),  # Δy=0.3 over 0.2s → 1.5
        ],
    )
    angles = AngleSequence(
        video="clip.mp4",
        frames=[
            _angle_frame(0, 0.00),
            _angle_frame(1, 0.10),
            _angle_frame(2, 0.30),
        ],
    )
    motion = compute_motion_derivatives(pose, angles, confidence_threshold=0.5)

    assert motion.frames[0].right_wrist_speed is None
    assert motion.frames[1].right_wrist_speed == pytest.approx(3.0)
    assert motion.frames[2].right_wrist_speed == pytest.approx(1.5)


def test_angular_velocity_uses_timestamps() -> None:
    pose = PoseSequence(
        video="clip.mp4",
        frames=[_pose_frame(i, i * 0.05) for i in range(3)],
    )
    angles = AngleSequence(
        video="clip.mp4",
        frames=[
            _angle_frame(0, 0.00, elbow=90.0, knee=170.0),
            _angle_frame(1, 0.05, elbow=100.0, knee=160.0),  # +200 deg/s, -200 deg/s
            _angle_frame(2, 0.15, elbow=110.0, knee=160.0),  # +100 deg/s over 0.1s
        ],
    )
    motion = compute_motion_derivatives(pose, angles, confidence_threshold=0.5)

    assert motion.frames[0].right_elbow_angular_velocity is None
    assert motion.frames[1].right_elbow_angular_velocity == pytest.approx(200.0)
    assert motion.frames[1].right_knee_angular_velocity == pytest.approx(-200.0)
    assert motion.frames[2].right_elbow_angular_velocity == pytest.approx(100.0)
    assert motion.frames[2].right_knee_angular_velocity == pytest.approx(0.0)


def test_missing_values_yield_null_without_interpolation() -> None:
    pose = PoseSequence(
        video="clip.mp4",
        frames=[
            _pose_frame(0, 0.0, wrist=(0.1, 0.1)),
            _pose_frame(1, 0.1, wrist=None),  # missing wrist
            _pose_frame(2, 0.2, wrist=(0.3, 0.1)),
        ],
    )
    angles = AngleSequence(
        video="clip.mp4",
        frames=[
            _angle_frame(0, 0.0, elbow=90.0, knee=100.0),
            _angle_frame(1, 0.1, elbow=None, knee=110.0),
            _angle_frame(2, 0.2, elbow=110.0, knee=120.0),
        ],
    )
    motion = compute_motion_derivatives(pose, angles, confidence_threshold=0.5)

    assert motion.frames[0].right_wrist_speed is None
    assert motion.frames[1].right_wrist_speed is None  # curr missing
    assert motion.frames[2].right_wrist_speed is None  # prev missing — no bridge

    assert motion.frames[1].right_elbow_angular_velocity is None
    assert motion.frames[2].right_elbow_angular_velocity is None  # prev angle null
    assert motion.frames[2].right_knee_angular_velocity == pytest.approx(100.0)


def test_low_confidence_wrist_treated_as_missing() -> None:
    pose = PoseSequence(
        video="clip.mp4",
        frames=[
            _pose_frame(0, 0.0, wrist=(0.0, 0.0), wrist_conf=0.9),
            _pose_frame(1, 0.1, wrist=(0.2, 0.0), wrist_conf=0.2),
        ],
    )
    angles = AngleSequence(
        video="clip.mp4",
        frames=[_angle_frame(0, 0.0), _angle_frame(1, 0.1)],
    )
    motion = compute_motion_derivatives(pose, angles, confidence_threshold=0.5)
    assert motion.frames[1].right_wrist_speed is None


def test_peaks_report_value_frame_and_timestamp() -> None:
    pose = PoseSequence(
        video="clip.mp4",
        frames=[
            _pose_frame(0, 0.0, wrist=(0.0, 0.0)),
            _pose_frame(1, 0.1, wrist=(0.1, 0.0)),  # speed 1.0
            _pose_frame(2, 0.2, wrist=(0.4, 0.0)),  # speed 3.0 peak
            _pose_frame(3, 0.3, wrist=(0.5, 0.0)),  # speed 1.0
        ],
    )
    angles = AngleSequence(
        video="clip.mp4",
        frames=[
            _angle_frame(0, 0.0, elbow=0.0, knee=0.0),
            _angle_frame(1, 0.1, elbow=10.0, knee=-5.0),
            _angle_frame(2, 0.2, elbow=15.0, knee=-25.0),  # knee |ω|=200 peak
            _angle_frame(3, 0.3, elbow=16.0, knee=-26.0),
        ],
    )
    motion = compute_motion_derivatives(pose, angles, confidence_threshold=0.5)

    wrist_peak = motion.peaks["right_wrist_speed"]
    assert wrist_peak.value == pytest.approx(3.0)
    assert wrist_peak.frame_index == 2
    assert wrist_peak.timestamp == pytest.approx(0.2)

    knee_peak = motion.peaks["right_knee_angular_velocity"]
    assert knee_peak.value == pytest.approx(200.0)  # magnitude
    assert knee_peak.frame_index == 2
    assert knee_peak.timestamp == pytest.approx(0.2)

    elbow_peak = motion.peaks["right_elbow_angular_velocity"]
    assert elbow_peak.value == pytest.approx(100.0)
    assert elbow_peak.frame_index == 1
