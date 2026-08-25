"""Phase-window stroke metrics (feature extraction only; no scoring)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.schemas.motion import PeakStats


@dataclass(slots=True)
class PhaseWindow:
    start_frame_index: int | None = None
    end_frame_index: int | None = None
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    duration: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration": self.duration,
        }


@dataclass(slots=True)
class PreparationMetrics:
    window: PhaseWindow = field(default_factory=PhaseWindow)
    mean_wrist_speed: float | None = None
    mean_elbow_angle: float | None = None
    mean_knee_angle: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.window.to_dict(),
            "mean_wrist_speed": self.mean_wrist_speed,
            "mean_elbow_angle": self.mean_elbow_angle,
            "mean_knee_angle": self.mean_knee_angle,
        }


@dataclass(slots=True)
class BackswingMetrics:
    window: PhaseWindow = field(default_factory=PhaseWindow)
    min_elbow_angle: PeakStats = field(
        default_factory=lambda: PeakStats(None, None, None)
    )
    peak_wrist_speed: PeakStats = field(
        default_factory=lambda: PeakStats(None, None, None)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.window.to_dict(),
            "min_elbow_angle": self.min_elbow_angle.to_dict(),
            "peak_wrist_speed": self.peak_wrist_speed.to_dict(),
        }


@dataclass(slots=True)
class AccelerationMetrics:
    window: PhaseWindow = field(default_factory=PhaseWindow)
    peak_wrist_speed: PeakStats = field(
        default_factory=lambda: PeakStats(None, None, None)
    )
    peak_elbow_angular_velocity: PeakStats = field(
        default_factory=lambda: PeakStats(None, None, None)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.window.to_dict(),
            "peak_wrist_speed": self.peak_wrist_speed.to_dict(),
            "peak_elbow_angular_velocity": self.peak_elbow_angular_velocity.to_dict(),
        }


@dataclass(slots=True)
class EstimatedContactMetrics:
    frame_index: int | None = None
    timestamp: float | None = None
    right_elbow_angle: float | None = None
    right_knee_angle: float | None = None
    right_wrist_speed: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "right_elbow_angle": self.right_elbow_angle,
            "right_knee_angle": self.right_knee_angle,
            "right_wrist_speed": self.right_wrist_speed,
        }


@dataclass(slots=True)
class FollowThroughMetrics:
    window: PhaseWindow = field(default_factory=PhaseWindow)
    mean_wrist_speed: float | None = None
    wrist_speed_at_end: float | None = None
    elbow_angle_at_end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.window.to_dict(),
            "mean_wrist_speed": self.mean_wrist_speed,
            "wrist_speed_at_end": self.wrist_speed_at_end,
            "elbow_angle_at_end": self.elbow_angle_at_end,
        }


@dataclass(slots=True)
class StrokeMetrics:
    video: str
    preparation: PreparationMetrics = field(default_factory=PreparationMetrics)
    backswing: BackswingMetrics = field(default_factory=BackswingMetrics)
    acceleration: AccelerationMetrics = field(default_factory=AccelerationMetrics)
    estimated_contact: EstimatedContactMetrics = field(
        default_factory=EstimatedContactMetrics
    )
    follow_through: FollowThroughMetrics = field(default_factory=FollowThroughMetrics)
    notes: str = (
        "Phase-window feature extraction from joint angles and motion. "
        "ESTIMATED_CONTACT uses the peak right-wrist-speed anchor; "
        "no scoring, technique feedback, or shuttle tracking."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "notes": self.notes,
            "preparation": self.preparation.to_dict(),
            "backswing": self.backswing.to_dict(),
            "acceleration": self.acceleration.to_dict(),
            "estimated_contact": self.estimated_contact.to_dict(),
            "follow_through": self.follow_through.to_dict(),
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
