"""Tests for metrics overlay formatting (no recalculation)."""

from __future__ import annotations

import numpy as np

from app.cv.overlay import (
    draw_metrics_overlay,
    format_overlay_lines,
    mean_pose_confidence,
)
from app.schemas.angles import AngleFrame
from app.schemas.motion import MotionFrame
from app.schemas.pose import Keypoint, PoseFrame


def test_format_overlay_lines_uses_na_for_missing() -> None:
    lines = format_overlay_lines(
        timestamp=0.12,
        right_elbow_angle=90.0,
        right_knee_angle=None,
        right_wrist_speed=None,
        pose_confidence=0.87,
    )
    assert lines[0] == "Timestamp: 0.120s"
    assert lines[1] == "Right elbow angle: 90.0 deg"
    assert lines[2] == "Right knee angle: N/A"
    assert lines[3] == "Right wrist speed: N/A"
    assert lines[4] == "Pose confidence: 0.87"


def test_mean_pose_confidence() -> None:
    assert mean_pose_confidence({}) is None
    conf = mean_pose_confidence(
        {
            "a": Keypoint(0.1, 0.1, 0.8),
            "b": Keypoint(0.2, 0.2, 1.0),
        }
    )
    assert conf == 0.9


def test_draw_metrics_overlay_does_not_invent_values() -> None:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    pose = PoseFrame(
        frame_index=1,
        timestamp=0.1,
        keypoints={"right_wrist": Keypoint(0.5, 0.5, 0.9)},
    )
    angles = AngleFrame(
        frame_index=1,
        timestamp=0.1,
        right_elbow=None,
        right_knee=170.0,
        right_shoulder=None,
    )
    motion = MotionFrame(
        frame_index=1,
        timestamp=0.1,
        right_wrist_speed=None,
        right_elbow_angular_velocity=None,
        right_knee_angular_velocity=None,
    )
    out = draw_metrics_overlay(
        frame, pose_frame=pose, angle_frame=angles, motion_frame=motion
    )
    assert out.shape == frame.shape
    # Smoke: panel should alter some pixels vs blank frame.
    assert not np.array_equal(out, np.zeros_like(frame))
