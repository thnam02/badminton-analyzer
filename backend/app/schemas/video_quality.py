"""Video suitability assessment for smash analysis (limitations only; no hard reject)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class VideoQualityMetrics:
    """Individual measurable quality signals."""

    fps: float | None = None
    width: int | None = None
    height: int | None = None
    full_body_coverage: float | None = None
    racket_arm_visibility: float | None = None
    mean_pose_confidence: float | None = None
    missing_keypoint_fraction: float | None = None
    interpolated_keypoint_fraction: float | None = None
    player_size_ratio: float | None = None
    camera_stability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VideoQualityReport:
    """Typed quality report for an uploaded clip.

    ``usable`` stays True unless the clip is empty / has no detectable pose.
    Warnings mark limitations; they do not block analysis.
    """

    video: str
    usable: bool = True
    analysis_confidence: float = 0.0
    metrics: VideoQualityMetrics = field(default_factory=VideoQualityMetrics)
    warnings: list[str] = field(default_factory=list)
    notes: str = (
        "Video quality assessment marks limitations only; "
        "clips are not rejected aggressively."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "usable": self.usable,
            "analysis_confidence": self.analysis_confidence,
            "metrics": self.metrics.to_dict(),
            "warnings": list(self.warnings),
            "notes": self.notes,
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
