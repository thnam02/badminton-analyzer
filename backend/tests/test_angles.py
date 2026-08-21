"""Unit tests for geometric joint-angle calculation."""

from __future__ import annotations

import math

import pytest

from app.processing.angles import (
    angle_at_vertex,
    compute_angle_sequence,
    compute_joint_angle,
)
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence


def _kp(x: float, y: float, confidence: float = 0.95) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=confidence)


def test_angle_at_vertex_right_angle() -> None:
    # Vertex at origin; proximal on +x, distal on +y → 90°.
    proximal = _kp(1.0, 0.0)
    vertex = _kp(0.0, 0.0)
    distal = _kp(0.0, 1.0)
    assert angle_at_vertex(proximal, vertex, distal) == pytest.approx(90.0)


def test_angle_at_vertex_straight_line() -> None:
    proximal = _kp(0.0, 0.0)
    vertex = _kp(0.5, 0.0)
    distal = _kp(1.0, 0.0)
    assert angle_at_vertex(proximal, vertex, distal) == pytest.approx(180.0)


def test_angle_at_vertex_acute() -> None:
    # 60°: unit vectors at 0° and 60°.
    proximal = _kp(1.0, 0.0)
    vertex = _kp(0.0, 0.0)
    distal = _kp(0.5, math.sqrt(3) / 2)
    assert angle_at_vertex(proximal, vertex, distal) == pytest.approx(60.0, abs=1e-6)


def test_right_elbow_known_geometry() -> None:
    # Shoulder (0,0), elbow (1,0), wrist (1,1) → 90° at elbow.
    keypoints = {
        "right_shoulder": _kp(0.0, 0.0),
        "right_elbow": _kp(1.0, 0.0),
        "right_wrist": _kp(1.0, 1.0),
    }
    assert compute_joint_angle(
        keypoints,
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        confidence_threshold=0.5,
    ) == pytest.approx(90.0)


def test_right_knee_known_geometry() -> None:
    keypoints = {
        "right_hip": _kp(0.2, 0.2),
        "right_knee": _kp(0.2, 0.5),
        "right_ankle": _kp(0.5, 0.5),
    }
    assert compute_joint_angle(
        keypoints,
        "right_hip",
        "right_knee",
        "right_ankle",
        confidence_threshold=0.5,
    ) == pytest.approx(90.0)


def test_right_shoulder_known_geometry() -> None:
    # Hip below shoulder, elbow to the right → 90° at shoulder.
    keypoints = {
        "right_hip": _kp(0.4, 0.8),
        "right_shoulder": _kp(0.4, 0.4),
        "right_elbow": _kp(0.7, 0.4),
    }
    assert compute_joint_angle(
        keypoints,
        "right_hip",
        "right_shoulder",
        "right_elbow",
        confidence_threshold=0.5,
    ) == pytest.approx(90.0)


def test_low_confidence_returns_null() -> None:
    keypoints = {
        "right_shoulder": _kp(0.0, 0.0, confidence=0.95),
        "right_elbow": _kp(1.0, 0.0, confidence=0.2),
        "right_wrist": _kp(1.0, 1.0, confidence=0.95),
    }
    assert (
        compute_joint_angle(
            keypoints,
            "right_shoulder",
            "right_elbow",
            "right_wrist",
            confidence_threshold=0.5,
        )
        is None
    )


def test_missing_keypoint_returns_null() -> None:
    keypoints = {
        "right_shoulder": _kp(0.0, 0.0),
        "right_elbow": _kp(1.0, 0.0),
    }
    assert (
        compute_joint_angle(
            keypoints,
            "right_shoulder",
            "right_elbow",
            "right_wrist",
            confidence_threshold=0.5,
        )
        is None
    )


def test_compute_angle_sequence_preserves_index_and_timestamp() -> None:
    seq = PoseSequence(
        video="clip_pose.mp4",
        frames=[
            PoseFrame(
                frame_index=3,
                timestamp=0.12,
                keypoints={
                    # Elbow 90°: shoulder–elbow–wrist
                    "right_shoulder": _kp(0.4, 0.4),
                    "right_elbow": _kp(0.7, 0.4),
                    "right_wrist": _kp(0.7, 0.7),
                    # Shoulder 90°: hip–shoulder–elbow
                    "right_hip": _kp(0.4, 0.8),
                    # Knee 90°: hip–knee–ankle
                    "right_knee": _kp(0.4, 0.5),
                    "right_ankle": _kp(0.7, 0.5),
                },
            ),
            PoseFrame(
                frame_index=4,
                timestamp=0.16,
                keypoints={
                    # Missing wrist → elbow null; shoulder/knee still ok if present.
                    "right_shoulder": _kp(0.4, 0.4),
                    "right_elbow": _kp(0.7, 0.4),
                    "right_hip": _kp(0.4, 0.8),
                    "right_knee": _kp(0.4, 0.5),
                    "right_ankle": _kp(0.7, 0.5),
                },
            ),
        ],
    )
    angles = compute_angle_sequence(seq, confidence_threshold=0.5)
    assert angles.frame_count == 2
    assert angles.frames[0].frame_index == 3
    assert angles.frames[0].timestamp == pytest.approx(0.12)
    assert angles.frames[0].right_elbow == pytest.approx(90.0)
    assert angles.frames[0].right_knee == pytest.approx(90.0)
    assert angles.frames[0].right_shoulder == pytest.approx(90.0)

    assert angles.frames[1].frame_index == 4
    assert angles.frames[1].timestamp == pytest.approx(0.16)
    assert angles.frames[1].right_elbow is None
    assert angles.frames[1].right_knee == pytest.approx(90.0)
    assert angles.frames[1].right_shoulder == pytest.approx(90.0)
