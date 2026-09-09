"""Tests for video quality assessment (limitations only; no hard reject)."""

from __future__ import annotations

import pytest

from app.processing.video_quality import assess_video_quality
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence

_COCO_COUNT = 17


def _kp(x: float, y: float, conf: float = 0.9) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=conf)


def _full_body_keypoints(*, conf: float = 0.9, scale: float = 0.55) -> dict[str, Keypoint]:
    """Build a COCO-17 skeleton filling ~``scale`` of frame height."""
    # Vertically centered stick figure.
    top = 0.5 - scale / 2
    mid = 0.5
    bot = 0.5 + scale / 2
    cx = 0.5
    return {
        "nose": _kp(cx, top + 0.02, conf),
        "left_eye": _kp(cx - 0.01, top + 0.01, conf),
        "right_eye": _kp(cx + 0.01, top + 0.01, conf),
        "left_ear": _kp(cx - 0.02, top + 0.02, conf),
        "right_ear": _kp(cx + 0.02, top + 0.02, conf),
        "left_shoulder": _kp(cx - 0.08, top + 0.12, conf),
        "right_shoulder": _kp(cx + 0.08, top + 0.12, conf),
        "left_elbow": _kp(cx - 0.12, mid - 0.05, conf),
        "right_elbow": _kp(cx + 0.12, mid - 0.05, conf),
        "left_wrist": _kp(cx - 0.14, mid + 0.02, conf),
        "right_wrist": _kp(cx + 0.14, mid + 0.02, conf),
        "left_hip": _kp(cx - 0.06, mid + 0.05, conf),
        "right_hip": _kp(cx + 0.06, mid + 0.05, conf),
        "left_knee": _kp(cx - 0.06, mid + 0.22, conf),
        "right_knee": _kp(cx + 0.06, mid + 0.22, conf),
        "left_ankle": _kp(cx - 0.06, bot - 0.02, conf),
        "right_ankle": _kp(cx + 0.06, bot - 0.02, conf),
    }


def _sequence(
    n: int = 20,
    *,
    fps: float = 30.0,
    conf: float = 0.9,
    scale: float = 0.55,
    drop_joints: set[str] | None = None,
    video: str = "clip.mp4",
) -> PoseSequence:
    drop = drop_joints or set()
    frames: list[PoseFrame] = []
    for i in range(n):
        kps = _full_body_keypoints(conf=conf, scale=scale)
        for name in drop:
            kps.pop(name, None)
        # Tiny stable torso motion (camera-stable).
        for name, kp in list(kps.items()):
            kps[name] = Keypoint(x=kp.x + i * 0.0002, y=kp.y, confidence=kp.confidence)
        frames.append(
            PoseFrame(
                frame_index=i,
                timestamp=i / fps if fps > 0 else float(i),
                keypoints=kps,
            )
        )
    return PoseSequence(video=video, frames=frames)


def test_good_clip_is_usable_with_high_confidence() -> None:
    pose = _sequence(fps=30.0, conf=0.92, scale=0.55)
    report = assess_video_quality(
        pose,
        smoothed_pose=pose,
        fps=30.0,
        width=1280,
        height=720,
        confidence_threshold=0.5,
    )
    assert report.usable is True
    assert report.analysis_confidence >= 0.7
    assert report.metrics.fps == pytest.approx(30.0)
    assert report.metrics.width == 1280
    assert report.metrics.height == 720
    assert report.metrics.full_body_coverage == pytest.approx(1.0)
    assert report.metrics.racket_arm_visibility == pytest.approx(1.0)
    assert report.metrics.mean_pose_confidence is not None
    assert report.metrics.mean_pose_confidence >= 0.9
    assert report.metrics.missing_keypoint_fraction == pytest.approx(0.0)
    assert report.metrics.player_size_ratio is not None
    assert report.metrics.player_size_ratio >= 0.4
    assert "LOW_FPS" not in report.warnings
    assert "LOW_POSE_CONFIDENCE" not in report.warnings
    payload = report.to_dict()
    assert "usable" in payload
    assert "analysis_confidence" in payload
    assert "metrics" in payload
    assert "warnings" in payload


def test_low_confidence_marks_warning_but_stays_usable() -> None:
    pose = _sequence(fps=30.0, conf=0.35, scale=0.55)
    report = assess_video_quality(
        pose,
        smoothed_pose=pose,
        fps=30.0,
        width=1280,
        height=720,
        confidence_threshold=0.5,
    )
    # Raw joints exist but sit below threshold → treated as missing; still usable.
    assert report.usable is True
    assert "LOW_POSE_CONFIDENCE" in report.warnings or "HIGH_MISSING_KEYPOINTS" in report.warnings
    assert report.analysis_confidence < 0.7
    assert report.metrics.mean_pose_confidence is not None
    assert report.metrics.mean_pose_confidence < 0.55


def test_low_fps_marks_warning_without_rejecting() -> None:
    pose = _sequence(fps=15.0, conf=0.9, scale=0.55)
    report = assess_video_quality(
        pose,
        smoothed_pose=pose,
        fps=15.0,
        width=1280,
        height=720,
        confidence_threshold=0.5,
    )
    assert report.usable is True
    assert "LOW_FPS" in report.warnings
    assert report.metrics.fps == pytest.approx(15.0)
    assert report.analysis_confidence > 0.0


def test_partially_missing_pose_marks_coverage_warnings() -> None:
    # Drop lower body + keep racket arm → partial pose.
    drop = {
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }
    pose = _sequence(fps=30.0, conf=0.9, scale=0.35, drop_joints=drop)
    report = assess_video_quality(
        pose,
        smoothed_pose=pose,
        fps=30.0,
        width=1280,
        height=720,
        confidence_threshold=0.5,
    )
    assert report.usable is True
    assert report.metrics.full_body_coverage == pytest.approx(0.0)
    assert report.metrics.racket_arm_visibility == pytest.approx(1.0)
    assert report.metrics.missing_keypoint_fraction is not None
    assert report.metrics.missing_keypoint_fraction > 0.2
    assert "LOW_FULL_BODY_COVERAGE" in report.warnings
    assert "HIGH_MISSING_KEYPOINTS" in report.warnings


def test_interpolated_fraction_from_raw_vs_smoothed() -> None:
    raw = _sequence(n=5, fps=30.0, conf=0.9)
    # Remove right_wrist on middle frames in raw.
    for i in (1, 2, 3):
        raw.frames[i].keypoints.pop("right_wrist", None)
    smoothed = _sequence(n=5, fps=30.0, conf=0.9)
    report = assess_video_quality(
        raw,
        smoothed_pose=smoothed,
        fps=30.0,
        width=1280,
        height=720,
        confidence_threshold=0.5,
    )
    assert report.metrics.interpolated_keypoint_fraction is not None
    # 3 frames × 1 joint / (5 × 17) ≈ 3/85
    expected = 3 / (5 * _COCO_COUNT)
    assert report.metrics.interpolated_keypoint_fraction == pytest.approx(expected)


def test_empty_pose_is_not_usable() -> None:
    report = assess_video_quality(
        PoseSequence(video="empty.mp4", frames=[]),
        fps=30.0,
        width=640,
        height=360,
    )
    assert report.usable is False
    assert "NO_POSE_DETECTED" in report.warnings
    assert report.analysis_confidence == pytest.approx(0.0)
