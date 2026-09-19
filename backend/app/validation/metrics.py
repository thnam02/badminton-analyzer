"""Metric helpers for pose validation (NPE, PCK, missing rate)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def normalized_pixel_error(
    pred_x: float,
    pred_y: float,
    gt_x: float,
    gt_y: float,
) -> float:
    """Euclidean distance in normalized image coordinates."""
    return float(math.hypot(pred_x - gt_x, pred_y - gt_y))


def mean_or_none(values: Sequence[float] | Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def pck_score(errors: Sequence[float], *, threshold: float) -> float | None:
    """Fraction of errors strictly below ``threshold`` (None if empty)."""
    if not errors:
        return None
    hits = sum(1 for e in errors if e < threshold)
    return float(hits / len(errors))


def missing_rate(*, annotated: int, detected: int) -> float | None:
    """Share of GT joints that were not detected (missing or below conf)."""
    if annotated <= 0:
        return None
    missing = max(0, annotated - detected)
    return float(missing / annotated)
