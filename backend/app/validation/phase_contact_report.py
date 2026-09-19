"""Versioned phase / contact timing validation report schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PHASE_CONTACT_VALIDATION_REPORT_VERSION = "1.0.0"


@dataclass(slots=True)
class BoundaryTimingError:
    boundary: str
    gt_frame_index: int
    gt_timestamp: float | None
    pred_frame_index: int | None
    pred_timestamp: float | None
    absolute_frame_error: int | None
    absolute_time_error_ms: float | None
    predicted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "gt_frame_index": self.gt_frame_index,
            "gt_timestamp": self.gt_timestamp,
            "pred_frame_index": self.pred_frame_index,
            "pred_timestamp": self.pred_timestamp,
            "absolute_frame_error": self.absolute_frame_error,
            "absolute_time_error_ms": self.absolute_time_error_ms,
            "predicted": self.predicted,
        }


@dataclass(slots=True)
class VideoTimingResult:
    video: str
    fps: float
    analysis_confidence: float | None
    contact_type: str | None
    contact_annotated: bool
    contact_predicted: bool
    boundary_errors: list[BoundaryTimingError] = field(default_factory=list)
    contact_frame_error: int | None = None
    contact_time_error_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "fps": self.fps,
            "analysis_confidence": self.analysis_confidence,
            "contact_type": self.contact_type,
            "contact_annotated": self.contact_annotated,
            "contact_predicted": self.contact_predicted,
            "contact_frame_error": self.contact_frame_error,
            "contact_time_error_ms": self.contact_time_error_ms,
            "boundary_errors": [b.to_dict() for b in self.boundary_errors],
        }


@dataclass(slots=True)
class TimingAggregateStats:
    name: str
    sample_count: int
    mae_frames: float | None
    median_frames: float | None
    p90_frames: float | None
    mae_ms: float | None
    median_ms: float | None
    p90_ms: float | None
    within_frame_tol: dict[str, float | None] = field(default_factory=dict)
    within_time_tol_ms: dict[str, float | None] = field(default_factory=dict)
    missing_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "mae_frames": self.mae_frames,
            "median_frames": self.median_frames,
            "p90_frames": self.p90_frames,
            "mae_ms": self.mae_ms,
            "median_ms": self.median_ms,
            "p90_ms": self.p90_ms,
            "within_frame_tol": self.within_frame_tol,
            "within_time_tol_ms": self.within_time_tol_ms,
            "missing_rate": self.missing_rate,
        }


@dataclass(slots=True)
class GroupTimingBreakdown:
    """Aggregate timing stats for a group key (contact type, FPS, confidence)."""

    key: str
    sample_count: int
    contact: TimingAggregateStats | None = None
    boundaries: dict[str, TimingAggregateStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sample_count": self.sample_count,
            "contact": self.contact.to_dict() if self.contact else None,
            "boundaries": {k: v.to_dict() for k, v in sorted(self.boundaries.items())},
        }


@dataclass(slots=True)
class PhaseContactValidationReport:
    """Versioned per-video + aggregate phase/contact timing validation."""

    phase_contact_validation_report_version: str = (
        PHASE_CONTACT_VALIDATION_REPORT_VERSION
    )
    annotation_version: str = ""
    video_count: int = 0
    overall_contact: TimingAggregateStats | None = None
    overall_boundaries: dict[str, TimingAggregateStats] = field(default_factory=dict)
    by_contact_type: list[GroupTimingBreakdown] = field(default_factory=list)
    by_fps: list[GroupTimingBreakdown] = field(default_factory=list)
    by_analysis_confidence: list[GroupTimingBreakdown] = field(default_factory=list)
    contact_missing_rate: float | None = None
    videos: list[VideoTimingResult] = field(default_factory=list)
    debug_image_paths: list[str] = field(default_factory=list)
    notes: str = (
        "Offline phase-boundary and contact timing validation. "
        "Does not modify the phase detector, contact resolver, or /analyze."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_contact_validation_report_version": (
                self.phase_contact_validation_report_version
            ),
            "annotation_version": self.annotation_version,
            "video_count": self.video_count,
            "contact_missing_rate": self.contact_missing_rate,
            "overall_contact": (
                self.overall_contact.to_dict() if self.overall_contact else None
            ),
            "overall_boundaries": {
                k: v.to_dict() for k, v in sorted(self.overall_boundaries.items())
            },
            "by_contact_type": [g.to_dict() for g in self.by_contact_type],
            "by_fps": [g.to_dict() for g in self.by_fps],
            "by_analysis_confidence": [
                g.to_dict() for g in self.by_analysis_confidence
            ],
            "videos": [v.to_dict() for v in self.videos],
            "debug_image_paths": list(self.debug_image_paths),
            "notes": self.notes,
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
