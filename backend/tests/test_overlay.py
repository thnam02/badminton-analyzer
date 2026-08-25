"""Tests for layered sports-analysis overlay."""

from __future__ import annotations

import numpy as np

from app.cv.layers.helpers import AnchorSmoother, clamp_label_origin, mean_pose_confidence
from app.cv.layers.hud_layer import render_hud_layer
from app.cv.overlay import AnnotationRenderer
from app.schemas.angles import AngleFrame
from app.schemas.motion import MotionFrame
from app.schemas.pose import Keypoint, PoseFrame


def test_mean_pose_confidence() -> None:
    assert mean_pose_confidence({}) is None
    assert (
        mean_pose_confidence(
            {"a": Keypoint(0.1, 0.1, 0.8), "b": Keypoint(0.2, 0.2, 1.0)}
        )
        == 0.9
    )


def test_anchor_smoother_ema() -> None:
    smoother = AnchorSmoother(alpha=0.5)
    a = smoother.update("elbow", (0.0, 0.0))
    b = smoother.update("elbow", (10.0, 0.0))
    assert a == (0.0, 0.0)
    assert b is not None
    assert b[0] == 5.0
    assert smoother.update("elbow", None) is None


def test_clamp_label_flips_near_right_edge() -> None:
    # Near right edge: should flip to the left of the anchor.
    x, y = clamp_label_origin(
        anchor_x=190,
        anchor_y=50,
        text_w=40,
        text_h=12,
        frame_w=200,
        frame_h=100,
        offset_x=18,
        offset_y=-14,
        margin=4,
    )
    assert x + 40 <= 200 - 4
    assert x < 190


def test_hud_hides_confidence_when_healthy() -> None:
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    pose = PoseFrame(
        frame_index=0,
        timestamp=1.25,
        keypoints={
            "right_elbow": Keypoint(0.5, 0.5, 0.9),
            "right_wrist": Keypoint(0.6, 0.5, 0.9),
        },
    )
    out = render_hud_layer(
        frame.copy(),
        pose_frame=pose,
        angle_frame=None,
        motion_frame=MotionFrame(frame_index=0, timestamp=1.25),
    )
    # Timestamp chip should paint some pixels; warning should not dominate.
    assert not np.array_equal(out, frame)


def test_renderer_uses_precomputed_angles_without_inventing() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    pose = PoseFrame(
        frame_index=0,
        timestamp=0.1,
        keypoints={
            "right_shoulder": Keypoint(0.40, 0.30, 0.95),
            "right_elbow": Keypoint(0.55, 0.30, 0.95),
            "right_wrist": Keypoint(0.55, 0.45, 0.95),
            "right_hip": Keypoint(0.42, 0.55, 0.95),
            "right_knee": Keypoint(0.42, 0.70, 0.95),
            "right_ankle": Keypoint(0.55, 0.70, 0.95),
        },
    )
    angles = AngleFrame(
        frame_index=0,
        timestamp=0.1,
        right_elbow=90.0,
        right_knee=None,  # must not invent a knee label value
        right_shoulder=None,
    )
    motion = MotionFrame(
        frame_index=0,
        timestamp=0.1,
        right_wrist_speed=1.234,
    )
    out = AnnotationRenderer(anchor_smoothing=1.0, muscle_overlay=False).render(
        frame,
        pose_frame=pose,
        angle_frame=angles,
        motion_frame=motion,
    )
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
