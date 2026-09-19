"""Aggregate error helpers for angle validation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def mean_absolute_error(errors: Sequence[float]) -> float | None:
    if not errors:
        return None
    return float(sum(abs(e) for e in errors) / len(errors))


def median_absolute_error(errors: Sequence[float]) -> float | None:
    if not errors:
        return None
    vals = sorted(abs(e) for e in errors)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def percentile_absolute_error(
    errors: Sequence[float],
    *,
    percentile: float,
) -> float | None:
    """Linear-interpolated percentile of absolute errors (percentile in 0–100)."""
    if not errors:
        return None
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    vals = sorted(abs(e) for e in errors)
    if len(vals) == 1:
        return float(vals[0])
    rank = (percentile / 100.0) * (len(vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(vals[lo])
    weight = rank - lo
    return float(vals[lo] * (1.0 - weight) + vals[hi] * weight)


def invalid_rate(*, annotated: int, valid_predictions: int) -> float | None:
    """Share of annotated angles with missing/invalid predictions."""
    if annotated <= 0:
        return None
    invalid = max(0, annotated - valid_predictions)
    return float(invalid / annotated)


def mean_or_none(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(sum(vals) / len(vals))
