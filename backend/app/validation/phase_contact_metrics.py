"""Timing-error helpers for phase / contact validation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def absolute_frame_error(pred_frame: int, gt_frame: int) -> int:
    return abs(int(pred_frame) - int(gt_frame))


def absolute_time_error_ms(
    pred_timestamp: float,
    gt_timestamp: float,
) -> float:
    """Absolute time error in milliseconds."""
    return abs(float(pred_timestamp) - float(gt_timestamp)) * 1000.0


def frame_error_to_ms(frame_error: int | float, *, fps: float) -> float | None:
    if fps <= 0:
        return None
    return float(abs(frame_error)) / float(fps) * 1000.0


def mean_or_none(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    vals = sorted(float(v) for v in values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def percentile_or_none(
    values: Sequence[float],
    *,
    percentile: float,
) -> float | None:
    if not values:
        return None
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return float(vals[0])
    rank = (percentile / 100.0) * (len(vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(vals[lo])
    weight = rank - lo
    return float(vals[lo] * (1.0 - weight) + vals[hi] * weight)


def within_tolerance_rate(
    errors: Sequence[float],
    *,
    tolerance: float,
) -> float | None:
    """Fraction of absolute errors ``<= tolerance`` (None if empty)."""
    if not errors:
        return None
    hits = sum(1 for e in errors if abs(float(e)) <= tolerance)
    return float(hits / len(errors))


def missing_rate(*, annotated: int, predicted: int) -> float | None:
    if annotated <= 0:
        return None
    missing = max(0, annotated - predicted)
    return float(missing / annotated)
