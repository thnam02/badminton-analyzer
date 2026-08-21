"""Temporal preprocessing over PoseSequence trajectories."""

from __future__ import annotations

import copy
from collections.abc import Iterable

import numpy as np
from scipy.signal import savgol_filter

from app.schemas.pose import Keypoint, PoseFrame, PoseSequence


def preprocess_pose_sequence(
    sequence: PoseSequence,
    *,
    confidence_threshold: float = 0.5,
    max_gap: int = 5,
    savgol_window: int = 7,
    savgol_polyorder: int = 2,
) -> PoseSequence:
    """Return a new smoothed sequence; ``sequence`` is left unchanged.

    Per joint, across frames:
    1. Drop samples below ``confidence_threshold`` (treat as missing)
    2. Linearly interpolate gaps of length ``<= max_gap``
    3. Savitzky–Golay smooth normalized x/y (confidence left as-is after interp)
    """
    if sequence.frame_count == 0:
        return PoseSequence(video=sequence.video, frames=[])

    # Work on a deep copy so the caller's raw sequence stays intact.
    working = copy.deepcopy(sequence)
    joint_names = _collect_joint_names(working.frames)
    n = working.frame_count

    # joint -> arrays of length n (NaN where missing / low confidence)
    xs: dict[str, np.ndarray] = {}
    ys: dict[str, np.ndarray] = {}
    cs: dict[str, np.ndarray] = {}

    for name in joint_names:
        x = np.full(n, np.nan, dtype=np.float64)
        y = np.full(n, np.nan, dtype=np.float64)
        c = np.full(n, np.nan, dtype=np.float64)
        for i, frame in enumerate(working.frames):
            kp = frame.keypoints.get(name)
            if kp is None or kp.confidence < confidence_threshold:
                continue
            x[i] = kp.x
            y[i] = kp.y
            c[i] = kp.confidence
        x = _interpolate_short_gaps(x, max_gap=max_gap)
        y = _interpolate_short_gaps(y, max_gap=max_gap)
        c = _interpolate_short_gaps(c, max_gap=max_gap)
        x = _savgol_nan_safe(x, window=savgol_window, polyorder=savgol_polyorder)
        y = _savgol_nan_safe(y, window=savgol_window, polyorder=savgol_polyorder)
        xs[name] = x
        ys[name] = y
        cs[name] = c

    smoothed_frames: list[PoseFrame] = []
    for i, frame in enumerate(working.frames):
        keypoints: dict[str, Keypoint] = {}
        for name in joint_names:
            if np.isnan(xs[name][i]) or np.isnan(ys[name][i]):
                continue
            conf = cs[name][i]
            if np.isnan(conf):
                conf = confidence_threshold
            keypoints[name] = Keypoint(
                x=float(xs[name][i]),
                y=float(ys[name][i]),
                confidence=float(conf),
            )
        smoothed_frames.append(
            PoseFrame(
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                keypoints=keypoints,
            )
        )

    return PoseSequence(video=sequence.video, frames=smoothed_frames)


def _collect_joint_names(frames: Iterable[PoseFrame]) -> list[str]:
    names: set[str] = set()
    for frame in frames:
        names.update(frame.keypoints.keys())
    return sorted(names)


def _interpolate_short_gaps(values: np.ndarray, *, max_gap: int) -> np.ndarray:
    """Linearly fill interior NaN runs of length <= max_gap; leave longer gaps."""
    out = values.copy()
    n = len(out)
    i = 0
    while i < n:
        if not np.isnan(out[i]):
            i += 1
            continue
        start = i
        while i < n and np.isnan(out[i]):
            i += 1
        end = i  # exclusive
        gap = end - start
        left = start - 1
        right = end
        if gap > max_gap:
            continue
        if left < 0 or right >= n:
            # Leading/trailing gaps: do not extrapolate.
            continue
        if np.isnan(out[left]) or np.isnan(out[right]):
            continue
        for g in range(gap):
            t = (g + 1) / (gap + 1)
            out[start + g] = (1.0 - t) * out[left] + t * out[right]
    return out


def _savgol_nan_safe(
    values: np.ndarray,
    *,
    window: int,
    polyorder: int,
) -> np.ndarray:
    """Apply Savitzky–Golay on contiguous finite segments only."""
    out = values.copy()
    n = len(out)
    if n == 0:
        return out

    window = int(window)
    polyorder = int(polyorder)
    if window % 2 == 0:
        window += 1
    if window < polyorder + 2:
        window = polyorder + 2 + (1 - (polyorder + 2) % 2)

    i = 0
    while i < n:
        if np.isnan(out[i]):
            i += 1
            continue
        start = i
        while i < n and not np.isnan(out[i]):
            i += 1
        end = i
        segment = out[start:end]
        seg_len = end - start
        if seg_len < window:
            # Too short for configured window: try the largest valid odd window.
            w = seg_len if seg_len % 2 == 1 else seg_len - 1
            if w >= polyorder + 2 and w >= 3:
                out[start:end] = savgol_filter(segment, window_length=w, polyorder=polyorder)
            continue
        out[start:end] = savgol_filter(segment, window_length=window, polyorder=polyorder)
    return out
