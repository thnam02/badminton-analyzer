"""Aggregated smash stroke metrics derived from pose / angles / motion / phases."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StrokeMetrics:
    """Phase-aware summary metrics for rule-based technique evaluation."""

    video: str
    estimated_contact_frame_index: int | None = None
    estimated_contact_timestamp: float | None = None
    phase_confidence: float = 0.0

    # Contact snapshot
    contact_elbow_angle_deg: float | None = None
    contact_knee_angle_deg: float | None = None
    contact_shoulder_angle_deg: float | None = None
    contact_wrist_y_normalized: float | None = None
    peak_wrist_speed: float | None = None

    # Knee contribution: knee extension from preparation to contact (degrees).
    knee_contribution_deg: float | None = None
    preparation_knee_angle_deg: float | None = None

    # Acceleration timing: frames from peak |elbow ω| to estimated contact.
    # Negative → peak before contact; positive → peak after contact.
    peak_elbow_omega_offset_frames: int | None = None
    acceleration_phase_fraction: float | None = None

    # Follow-through: speed retention and duration after contact.
    follow_through_speed_ratio: float | None = None
    follow_through_frame_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
