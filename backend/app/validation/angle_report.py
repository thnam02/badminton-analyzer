"""Versioned angle validation report schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ANGLE_VALIDATION_REPORT_VERSION = "1.0.0"


@dataclass(slots=True)
class AngleErrorDetail:
    angle_name: str
    gt_degrees: float
    pred_degrees: float | None
    absolute_error: float | None
    predicted: bool
    pose_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle_name": self.angle_name,
            "gt_degrees": self.gt_degrees,
            "pred_degrees": self.pred_degrees,
            "absolute_error": self.absolute_error,
            "predicted": self.predicted,
            "pose_confidence": self.pose_confidence,
        }


@dataclass(slots=True)
class FrameAngleValidationResult:
    frame_index: int
    phase: str | None
    pose_confidence: float | None
    angle_errors: list[AngleErrorDetail] = field(default_factory=list)
    mean_absolute_error: float | None = None
    annotated_angle_count: int = 0
    valid_prediction_count: int = 0
    invalid_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "phase": self.phase,
            "pose_confidence": self.pose_confidence,
            "mean_absolute_error": self.mean_absolute_error,
            "annotated_angle_count": self.annotated_angle_count,
            "valid_prediction_count": self.valid_prediction_count,
            "invalid_rate": self.invalid_rate,
            "angle_errors": [e.to_dict() for e in self.angle_errors],
        }


@dataclass(slots=True)
class AggregateAngleStats:
    name: str
    sample_count: int
    mae: float | None
    median_absolute_error: float | None
    p90_absolute_error: float | None
    invalid_rate: float | None
    per_angle: list["AggregateAngleStats"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "mae": self.mae,
            "median_absolute_error": self.median_absolute_error,
            "p90_absolute_error": self.p90_absolute_error,
            "invalid_rate": self.invalid_rate,
            "per_angle": [a.to_dict() for a in self.per_angle],
        }


@dataclass(slots=True)
class PhaseAngleBreakdown:
    phase: str
    frame_count: int
    sample_count: int
    mae: float | None
    median_absolute_error: float | None
    p90_absolute_error: float | None
    invalid_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "frame_count": self.frame_count,
            "sample_count": self.sample_count,
            "mae": self.mae,
            "median_absolute_error": self.median_absolute_error,
            "p90_absolute_error": self.p90_absolute_error,
            "invalid_rate": self.invalid_rate,
        }


@dataclass(slots=True)
class ConfidenceBinBreakdown:
    """Error stats for samples whose pose confidence falls in ``[lo, hi)``."""

    label: str
    confidence_min: float
    confidence_max: float
    sample_count: int
    mae: float | None
    median_absolute_error: float | None
    p90_absolute_error: float | None
    invalid_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence_min": self.confidence_min,
            "confidence_max": self.confidence_max,
            "sample_count": self.sample_count,
            "mae": self.mae,
            "median_absolute_error": self.median_absolute_error,
            "p90_absolute_error": self.p90_absolute_error,
            "invalid_rate": self.invalid_rate,
        }


@dataclass(slots=True)
class AngleValidationReport:
    """Versioned summary + per-frame records for offline angle validation."""

    video: str
    angle_validation_report_version: str = ANGLE_VALIDATION_REPORT_VERSION
    annotation_version: str = ""
    annotated_frame_count: int = 0
    matched_frame_count: int = 0
    overall: AggregateAngleStats | None = None
    by_phase: list[PhaseAngleBreakdown] = field(default_factory=list)
    by_confidence: list[ConfidenceBinBreakdown] = field(default_factory=list)
    frames: list[FrameAngleValidationResult] = field(default_factory=list)
    debug_image_paths: list[str] = field(default_factory=list)
    notes: str = (
        "Offline angle validation against manual annotations / derived GT. "
        "Does not modify smoothing, angle formulas, or the production analyze path."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle_validation_report_version": self.angle_validation_report_version,
            "annotation_version": self.annotation_version,
            "video": self.video,
            "annotated_frame_count": self.annotated_frame_count,
            "matched_frame_count": self.matched_frame_count,
            "overall": self.overall.to_dict() if self.overall else None,
            "by_phase": [p.to_dict() for p in self.by_phase],
            "by_confidence": [c.to_dict() for c in self.by_confidence],
            "frames": [f.to_dict() for f in self.frames],
            "debug_image_paths": list(self.debug_image_paths),
            "notes": self.notes,
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
