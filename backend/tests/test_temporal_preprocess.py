"""Tests for temporal pose preprocessing."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from app.processing.temporal import (
    _interpolate_short_gaps,
    preprocess_pose_sequence,
)
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence


def _sequence_from_xy(
    xs: list[float | None],
    ys: list[float | None],
    *,
    joint: str = "right_wrist",
    confidence: float = 0.9,
) -> PoseSequence:
    frames: list[PoseFrame] = []
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        keypoints: dict[str, Keypoint] = {}
        if x is not None and y is not None:
            keypoints[joint] = Keypoint(x=float(x), y=float(y), confidence=confidence)
        frames.append(
            PoseFrame(frame_index=i, timestamp=float(i) / 30.0, keypoints=keypoints)
        )
    return PoseSequence(video="test_pose.mp4", frames=frames)


def test_short_gap_linear_interpolation_fills_missing_keypoint() -> None:
    # Frames 0..4 with a 1-frame hole at index 2 → expect midpoint 0.4.
    raw = _sequence_from_xy(
        [0.0, 0.2, None, 0.6, 0.8],
        [0.0, 0.2, None, 0.6, 0.8],
    )
    snapshot = copy.deepcopy(raw)

    smoothed = preprocess_pose_sequence(
        raw,
        confidence_threshold=0.5,
        max_gap=3,
        savgol_window=3,
        savgol_polyorder=1,
    )

    assert "right_wrist" not in raw.frames[2].keypoints
    assert snapshot.to_dict() == raw.to_dict()

    filled = smoothed.frames[2].keypoints["right_wrist"]
    assert filled.x == pytest.approx(0.4, abs=0.05)
    assert filled.y == pytest.approx(0.4, abs=0.05)


def test_long_gap_is_not_interpolated() -> None:
    raw = _sequence_from_xy(
        [0.0, None, None, None, None, 1.0],
        [0.0, None, None, None, None, 1.0],
    )
    smoothed = preprocess_pose_sequence(
        raw,
        confidence_threshold=0.5,
        max_gap=2,
        savgol_window=3,
        savgol_polyorder=1,
    )
    for i in range(1, 5):
        assert "right_wrist" not in smoothed.frames[i].keypoints


def test_low_confidence_treated_as_missing_then_interpolated() -> None:
    frames = []
    for i, x in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        conf = 0.2 if i == 2 else 0.95
        frames.append(
            PoseFrame(
                frame_index=i,
                timestamp=i / 30.0,
                keypoints={"right_wrist": Keypoint(x=x, y=x, confidence=conf)},
            )
        )
    raw = PoseSequence(video="test.mp4", frames=frames)
    smoothed = preprocess_pose_sequence(
        raw,
        confidence_threshold=0.5,
        max_gap=2,
        savgol_window=3,
        savgol_polyorder=1,
    )
    assert smoothed.frames[2].keypoints["right_wrist"].x == pytest.approx(0.5, abs=0.05)


def test_savgol_reduces_jitter() -> None:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 61)
    clean = 0.4 + 0.2 * np.sin(2 * np.pi * t)
    noisy = clean + rng.normal(0.0, 0.04, size=t.shape)

    raw = _sequence_from_xy(list(noisy), list(noisy))
    smoothed = preprocess_pose_sequence(
        raw,
        confidence_threshold=0.5,
        max_gap=2,
        savgol_window=11,
        savgol_polyorder=2,
    )

    raw_x = np.array([f.keypoints["right_wrist"].x for f in raw.frames])
    sm_x = np.array([f.keypoints["right_wrist"].x for f in smoothed.frames])

    raw_jitter = float(np.std(np.diff(raw_x)))
    sm_jitter = float(np.std(np.diff(sm_x)))
    assert sm_jitter < raw_jitter * 0.75


def test_interpolate_short_gaps_helper() -> None:
    values = np.array([0.0, np.nan, np.nan, 3.0], dtype=np.float64)
    filled = _interpolate_short_gaps(values, max_gap=2)
    assert filled[1] == pytest.approx(1.0)
    assert filled[2] == pytest.approx(2.0)

    long_gap = np.array([0.0, np.nan, np.nan, np.nan, 4.0], dtype=np.float64)
    still = _interpolate_short_gaps(long_gap, max_gap=2)
    assert np.isnan(still[1]) and np.isnan(still[2]) and np.isnan(still[3])
