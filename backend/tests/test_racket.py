"""Tests for independent racket detection / lightweight tracking."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.cv.racket.base import RawRacketDetection
from app.cv.racket.convert import detections_to_trajectory
from app.cv.racket.debug_render import render_racket_debug_video
from app.cv.racket.factory import get_racket_detector
from app.cv.racket.pose_context import infer_hitting_hand, load_pose_sequence
from app.cv.racket.pose_guided import PoseGuidedRacketDetector
from app.cv.racket.track import apply_lightweight_tracking
from app.cv.racket.yolo import YoloRacketDetector
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence
from app.schemas.racket import RacketBBox, RacketPoint, RacketTrajectory
from app.services.racket_service import RacketService
from app.services.video_service import (
    racket_debug_video_path_for,
    racket_json_path_for,
)


def _pose_right_arm(n: int = 10, dt: float = 1 / 30) -> PoseSequence:
    frames = []
    for i in range(n):
        # Elbow left of wrist → shaft extends further right.
        frames.append(
            PoseFrame(
                frame_index=i,
                timestamp=i * dt,
                keypoints={
                    "right_elbow": Keypoint(0.40, 0.50, 0.9),
                    "right_wrist": Keypoint(0.55 + i * 0.01, 0.48, 0.95),
                    "right_shoulder": Keypoint(0.35, 0.45, 0.9),
                    "left_wrist": Keypoint(0.20, 0.55, 0.2),
                },
            )
        )
    return PoseSequence(video="arm.mp4", frames=frames)


def _write_arm_video(
    path: Path,
    n: int = 10,
    size: tuple[int, int] = (160, 120),
) -> None:
    """Dark frame with a bright shaft beyond the right wrist (pose-guided)."""
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    assert writer.isOpened()
    try:
        for i in range(n):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            wx = int((0.55 + i * 0.01) * w)
            wy = int(0.48 * h)
            tip_x = min(w - 2, wx + 28)
            tip_y = wy - 4
            cv2.line(frame, (wx, wy), (tip_x, tip_y), (220, 220, 220), 3)
            writer.write(frame)
    finally:
        writer.release()


def test_infer_hitting_hand_prefers_right_when_stronger() -> None:
    pose = _pose_right_arm()
    assert infer_hitting_hand(pose) == "RIGHT"
    assert infer_hitting_hand(pose, preferred="LEFT") == "LEFT"


def test_detections_to_trajectory_normalized() -> None:
    dets = [
        RawRacketDetection(0, 80.0, 60.0, 70.0, 50.0, 100.0, 80.0, 0.8, "RIGHT"),
        RawRacketDetection(1, None, None, None, None, None, None, 0.0, "RIGHT"),
    ]
    traj = detections_to_trajectory(
        dets,
        video_path=Path("clip.mp4"),
        backend="pose_guided",
        fps=30.0,
        width=160,
        height=120,
        hitting_hand="RIGHT",
    )
    assert traj.hitting_hand == "RIGHT"
    assert traj.frames[0].x == pytest.approx(0.5)
    assert traj.frames[0].bbox is not None
    assert traj.frames[0].bbox.x1 == pytest.approx(70 / 160)
    assert traj.missing_frame_indices == [1]


def test_lightweight_tracking_fills_short_gaps() -> None:
    frames = [
        RacketPoint(0, 0.0, 0.1, 0.2, RacketBBox(0.05, 0.15, 0.15, 0.25), True, 0.9, "RIGHT"),
        RacketPoint(1, 0.1, None, None, None, False, 0.0, "RIGHT"),
        RacketPoint(2, 0.2, None, None, None, False, 0.0, "RIGHT"),
        RacketPoint(3, 0.3, 0.4, 0.5, RacketBBox(0.35, 0.45, 0.45, 0.55), True, 0.9, "RIGHT"),
        RacketPoint(4, 0.4, None, None, None, False, 0.0, "RIGHT"),
        RacketPoint(5, 0.5, None, None, None, False, 0.0, "RIGHT"),
        RacketPoint(6, 0.6, None, None, None, False, 0.0, "RIGHT"),
        RacketPoint(7, 0.7, None, None, None, False, 0.0, "RIGHT"),
        RacketPoint(8, 0.8, 0.7, 0.8, None, True, 0.9, "RIGHT"),
    ]
    traj = RacketTrajectory(
        video="x.mp4",
        backend="test",
        fps=10.0,
        width=100,
        height=100,
        hitting_hand="RIGHT",
        frames=frames,
        missing_frame_indices=[1, 2, 4, 5, 6, 7],
    )
    tracked = apply_lightweight_tracking(traj, max_gap=3, max_jump=1.0)
    assert tracked.frames[1].interpolated is True
    assert tracked.frames[1].tracked is True
    assert tracked.frames[1].x == pytest.approx(0.2)
    assert tracked.frames[4].visible is False


def test_racket_paths() -> None:
    pose = Path("/out/abc_pose.mp4")
    assert racket_json_path_for(pose).name == "abc_racket.json"
    assert racket_debug_video_path_for(pose).name == "abc_racket_debug.mp4"


def test_yolo_unavailable_without_weights() -> None:
    assert YoloRacketDetector(weights="").is_available() is False


def test_pose_guided_and_debug_export(tmp_path: Path) -> None:
    video = tmp_path / "arm.mp4"
    _write_arm_video(video)
    pose = _pose_right_arm(n=10)
    pose_json = tmp_path / "pose_smoothed.json"
    pose.save_json(pose_json)
    loaded = load_pose_sequence(pose_json)
    assert loaded.frame_count == 10

    detector = PoseGuidedRacketDetector()
    assert detector.is_available()
    dets = detector.detect(video, pose=loaded, hitting_hand="RIGHT")
    assert len(dets) == 10
    visible = [d for d in dets if d.confidence > 0]
    assert len(visible) >= 1

    traj = detections_to_trajectory(
        dets,
        video_path=video,
        backend="pose_guided",
        fps=30.0,
        width=160,
        height=120,
        hitting_hand="RIGHT",
    )
    traj = apply_lightweight_tracking(traj, max_gap=3)
    json_path = tmp_path / "racket.json"
    traj.save_json(json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["hitting_hand"] == "RIGHT"
    assert "frames" in payload and "bbox" in payload["frames"][0]

    debug = tmp_path / "racket_debug.mp4"
    render_racket_debug_video(video, traj, debug, trail_length=6)
    assert debug.is_file() and debug.stat().st_size > 0


def test_racket_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "racket_backend", "pose_guided")
    monkeypatch.setattr(settings, "racket_track_enabled", True)
    monkeypatch.setattr(settings, "racket_track_max_gap", 3)
    monkeypatch.setattr(settings, "racket_debug_trail_length", 6)

    video = tmp_path / "arm.mp4"
    _write_arm_video(video)
    pose = _pose_right_arm(n=10)
    output_stem = tmp_path / "run_pose.mp4"
    json_path, debug_path, traj = RacketService().track_video(
        video,
        output_stem,
        pose=pose,
        backend="pose_guided",
        hitting_hand="RIGHT",
    )
    assert json_path.name == "run_racket.json"
    assert debug_path.name == "run_racket_debug.mp4"
    assert json_path.is_file() and debug_path.is_file()
    assert traj.backend == "pose_guided"
    assert isinstance(get_racket_detector("pose_guided"), PoseGuidedRacketDetector)
