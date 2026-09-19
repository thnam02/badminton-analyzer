"""Binary classification metrics for technique-issue validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ConfusionCounts:
    """2x2 confusion counts for one issue (certain labels only)."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    @property
    def support(self) -> int:
        """Number of GT-positive certain samples (TP + FN)."""
        return self.true_positive + self.false_negative

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

    def add(
        self,
        *,
        gt_present: bool,
        pred_present: bool,
    ) -> None:
        if gt_present and pred_present:
            self.true_positive += 1
        elif (not gt_present) and pred_present:
            self.false_positive += 1
        elif gt_present and (not pred_present):
            self.false_negative += 1
        else:
            self.true_negative += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "support": self.support,
            "total": self.total,
        }


def precision(tp: int, fp: int) -> float | None:
    denom = tp + fp
    if denom <= 0:
        return None
    return float(tp) / float(denom)


def recall(tp: int, fn: int) -> float | None:
    denom = tp + fn
    if denom <= 0:
        return None
    return float(tp) / float(denom)


def specificity(tn: int, fp: int) -> float | None:
    denom = tn + fp
    if denom <= 0:
        return None
    return float(tn) / float(denom)


def f1_score(prec: float | None, rec: float | None) -> float | None:
    if prec is None or rec is None:
        return None
    denom = prec + rec
    if denom <= 0:
        return None
    return float(2.0 * prec * rec / denom)


def false_positive_rate(fp: int, tn: int) -> float | None:
    denom = fp + tn
    if denom <= 0:
        return None
    return float(fp) / float(denom)


def false_negative_rate(fn: int, tp: int) -> float | None:
    denom = fn + tp
    if denom <= 0:
        return None
    return float(fn) / float(denom)


def metrics_from_confusion(counts: ConfusionCounts) -> dict[str, Any]:
    """Compute precision / recall / F1 / specificity / FPR / FNR from counts."""
    prec = precision(counts.true_positive, counts.false_positive)
    rec = recall(counts.true_positive, counts.false_negative)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1_score(prec, rec),
        "specificity": specificity(counts.true_negative, counts.false_positive),
        "false_positive_rate": false_positive_rate(
            counts.false_positive, counts.true_negative
        ),
        "false_negative_rate": false_negative_rate(
            counts.false_negative, counts.true_positive
        ),
        "support": counts.support,
        "sample_count": counts.total,
        "confusion_matrix": counts.to_dict(),
    }
