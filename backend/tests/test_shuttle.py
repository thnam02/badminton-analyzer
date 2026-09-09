"""Tests for independent shuttle tracking (no pose / contact coupling)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.cv.shuttle.base import RawShuttleDetection
from app.cv.shuttle.convert import detections_to_trajectory
from app.cv.shuttle.debug_render import render_shuttle_debug_video
from app.cv.shuttle.factory import get_shuttle_tracker
from app.cv.shuttle.heuristic import HeuristicShuttleTracker
from app.cv.shuttle.tracknet_v3 import TrackNetV3Adapter, parse_tracknet_csv
from app.processing.shuttle_smooth import interpolate_short_gaps
from app.schemas.shuttle import ShuttlePoint, ShuttleTrajectory
from app.services.shuttle_service import ShuttleService
from app.services.video_service import (
    shuttle_debug_video_path_for,
    shuttle_json_path_for,
)


def _write_blob_video(path: Path, n: int = 12, size: tuple[int, int] = (160, 120)) -> None:
    """Synthetic clip with a moving white dot (heuristic-friendly)."""
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    assert writer.isOpened()
    try:
        for i in range(n):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            x = 20 + i * 8
            y = 40 + (i % 3) * 2
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_parse_tracknet_csv_to_raw(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_ball.csv"
    csv_path.write_text(
        "Frame,Visibility,X,Y\n"
        "0,1,80.0,60.0\n"
        "1,0,0,0\n"
        "2,1,90.5,61.0\n",
        encoding="utf-8",
    )
    dets = parse_tracknet_csv(csv_path)
    assert len(dets) == 3
    assert dets[0].visibility == 1 and dets[0].x_px == 80.0
    assert dets[1].visibility == 0 and dets[1].x_px is None
    assert dets[2].confidence == 1.0


def test_detections_to_trajectory_normalized() -> None:
    dets = [
        RawShuttleDetection(0, 80.0, 60.0, 1, 1.0),
        RawShuttleDetection(1, None, None, 0, 0.0),
        RawShuttleDetection(2, 160.0, 120.0, 1, 0.9),
    ]
    traj = detections_to_trajectory(
        dets,
        video_path=Path("clip.mp4"),
        backend="tracknetv3",
        fps=30.0,
        width=160,
        height=120,
    )
    assert traj.frames[0].x == pytest.approx(0.5)
    assert traj.frames[0].y == pytest.approx(0.5)
    assert traj.frames[0].visible is True
    assert traj.frames[1].visible is False
    assert traj.missing_frame_indices == [1]
    assert traj.frames[2].x == pytest.approx(1.0)
    assert traj.frames[2].timestamp == pytest.approx(2 / 30.0)


def test_interpolate_short_gaps_only() -> None:
    frames = [
        ShuttlePoint(0, 0.0, 0.1, 0.2, True, 1.0),
        ShuttlePoint(1, 0.1, None, None, False, 0.0),
        ShuttlePoint(2, 0.2, None, None, False, 0.0),
        ShuttlePoint(3, 0.3, 0.4, 0.5, True, 1.0),
        # Long gap (4 frames) should stay missing when max_gap=3
        ShuttlePoint(4, 0.4, None, None, False, 0.0),
        ShuttlePoint(5, 0.5, None, None, False, 0.0),
        ShuttlePoint(6, 0.6, None, None, False, 0.0),
        ShuttlePoint(7, 0.7, None, None, False, 0.0),
        ShuttlePoint(8, 0.8, 0.7, 0.8, True, 1.0),
    ]
    traj = ShuttleTrajectory(
        video="x.mp4",
        backend="test",
        fps=10.0,
        width=100,
        height=100,
        frames=frames,
        missing_frame_indices=[1, 2, 4, 5, 6, 7],
    )
    filled = interpolate_short_gaps(traj, max_gap=3)
    assert filled.frames[1].interpolated is True
    assert filled.frames[1].x == pytest.approx(0.2)
    assert filled.frames[2].y == pytest.approx(0.4)
    assert filled.frames[4].visible is False
    assert 4 in filled.missing_frame_indices
    assert 1 not in filled.missing_frame_indices


def test_shuttle_json_paths() -> None:
    pose = Path("/out/abc_pose.mp4")
    assert shuttle_json_path_for(pose).name == "abc_shuttle.json"
    assert shuttle_debug_video_path_for(pose).name == "abc_shuttle_debug.mp4"


def test_tracknet_unavailable_without_weights() -> None:
    adapter = TrackNetV3Adapter(root="", tracknet_weights="")
    assert adapter.is_available() is False


def test_heuristic_and_debug_export(tmp_path: Path) -> None:
    video = tmp_path / "dot.mp4"
    _write_blob_video(video)
    tracker = HeuristicShuttleTracker()
    assert tracker.is_available()
    dets = tracker.track(video)
    assert len(dets) == 12
    traj = detections_to_trajectory(
        dets,
        video_path=video,
        backend=tracker.name,
        fps=30.0,
        width=160,
        height=120,
    )
    traj = interpolate_short_gaps(traj, max_gap=3)
    json_path = tmp_path / "shuttle.json"
    traj.save_json(json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "frames" in payload and "missing_frame_indices" in payload
    assert payload["backend"] == "heuristic"

    debug = tmp_path / "shuttle_debug.mp4"
    render_shuttle_debug_video(video, traj, debug, trail_length=8)
    assert debug.is_file() and debug.stat().st_size > 0


def test_shuttle_service_heuristic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "shuttle_backend", "heuristic")
    monkeypatch.setattr(settings, "shuttle_interp_max_gap", 3)
    monkeypatch.setattr(settings, "shuttle_debug_trail_length", 8)

    video = tmp_path / "dot.mp4"
    _write_blob_video(video)
    output_stem = tmp_path / "run_pose.mp4"
    # Pose video may not exist; path helpers only need the name.
    json_path, debug_path, traj = ShuttleService().track_video(
        video,
        output_stem,
        backend="heuristic",
    )
    assert json_path.name == "run_shuttle.json"
    assert debug_path.name == "run_shuttle_debug.mp4"
    assert json_path.is_file() and debug_path.is_file()
    assert traj.backend == "heuristic"
    assert isinstance(get_shuttle_tracker("heuristic"), HeuristicShuttleTracker)
