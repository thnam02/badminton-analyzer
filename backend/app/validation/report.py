"""Versioned pose validation report schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

POSE_VALIDATION_REPORT_VERSION = "1.0.0"


@dataclass(slots=True)
class JointErrorDetail:
    joint: str
    error: float | None
    predicted: bool
    confidence: float | None
    gt_x: float
    gt_y: float
    pred_x: float | None = None
    pred_y: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "error": self.error,
            "predicted": self.predicted,
            "confidence": self.confidence,
            "gt": {"x": self.gt_x, "y": self.gt_y},
            "pred": (
                {"x": self.pred_x, "y": self.pred_y}
                if self.pred_x is not None and self.pred_y is not None
                else None
            ),
        }


@dataclass(slots=True)
class FrameValidationResult:
    frame_index: int
    phase: str | None
    joint_errors: list[JointErrorDetail] = field(default_factory=list)
    mean_error: float | None = None
    mean_confidence: float | None = None
    missing_detection_rate: float | None = None
    annotated_joint_count: int = 0
    detected_joint_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "phase": self.phase,
            "mean_error": self.mean_error,
            "mean_confidence": self.mean_confidence,
            "missing_detection_rate": self.missing_detection_rate,
            "annotated_joint_count": self.annotated_joint_count,
            "detected_joint_count": self.detected_joint_count,
            "joint_errors": [j.to_dict() for j in self.joint_errors],
        }


@dataclass(slots=True)
class AggregateJointStats:
    joint: str
    sample_count: int
    mean_error: float | None
    mean_confidence: float | None
    missing_detection_rate: float | None
    pck: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "sample_count": self.sample_count,
            "mean_error": self.mean_error,
            "mean_confidence": self.mean_confidence,
            "missing_detection_rate": self.missing_detection_rate,
            "pck": self.pck,
        }


@dataclass(slots=True)
class AggregateGroupStats:
    """Aggregate over a joint group (all COCO-17 or badminton-critical)."""

    name: str
    sample_count: int
    mean_error: float | None
    mean_confidence: float | None
    missing_detection_rate: float | None
    pck: dict[str, float | None] = field(default_factory=dict)
    per_joint: list[AggregateJointStats] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "mean_error": self.mean_error,
            "mean_confidence": self.mean_confidence,
            "missing_detection_rate": self.missing_detection_rate,
            "pck": self.pck,
            "per_joint": [j.to_dict() for j in self.per_joint],
        }


@dataclass(slots=True)
class PhaseBreakdown:
    phase: str
    frame_count: int
    mean_error: float | None
    mean_confidence: float | None
    missing_detection_rate: float | None
    pck: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "frame_count": self.frame_count,
            "mean_error": self.mean_error,
            "mean_confidence": self.mean_confidence,
            "missing_detection_rate": self.missing_detection_rate,
            "pck": self.pck,
        }


@dataclass(slots=True)
class PoseValidationReport:
    """Versioned summary + per-frame errors for offline pose validation."""

    video: str
    pose_validation_report_version: str = POSE_VALIDATION_REPORT_VERSION
    annotation_version: str = ""
    pck_thresholds: list[float] = field(default_factory=list)
    confidence_threshold: float = 0.0
    annotated_frame_count: int = 0
    matched_frame_count: int = 0
    overall: AggregateGroupStats | None = None
    badminton_critical: AggregateGroupStats | None = None
    by_phase: list[PhaseBreakdown] = field(default_factory=list)
    frames: list[FrameValidationResult] = field(default_factory=list)
    debug_image_paths: list[str] = field(default_factory=list)
    notes: str = (
        "Offline RTMPose validation against manual COCO-17 annotations. "
        "Does not modify inference, smoothing, or the production analyze path."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose_validation_report_version": self.pose_validation_report_version,
            "annotation_version": self.annotation_version,
            "video": self.video,
            "pck_thresholds": list(self.pck_thresholds),
            "confidence_threshold": self.confidence_threshold,
            "annotated_frame_count": self.annotated_frame_count,
            "matched_frame_count": self.matched_frame_count,
            "overall": self.overall.to_dict() if self.overall else None,
            "badminton_critical": (
                self.badminton_critical.to_dict() if self.badminton_critical else None
            ),
            "by_phase": [p.to_dict() for p in self.by_phase],
            "frames": [f.to_dict() for f in self.frames],
            "debug_image_paths": list(self.debug_image_paths),
            "notes": self.notes,
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
