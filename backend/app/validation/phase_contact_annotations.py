"""Manual annotations for phase-boundary and contact timing validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.validation.phase_contact_boundaries import BOUNDARY_NAMES

PHASE_CONTACT_VALIDATION_ANNOTATION_VERSION = "1.0.0"
_DEFAULT_NOTES = (
    "Ground-truth smash phase boundaries and contact frame/timestamp. "
    "Used only by offline PhaseContactValidator — not by /analyze."
)


@dataclass(slots=True)
class BoundaryAnnotation:
    """One labeled boundary (frame index required; timestamp optional)."""

    frame_index: int
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"frame_index": int(self.frame_index)}
        if self.timestamp is not None:
            payload["timestamp"] = float(self.timestamp)
        return payload


@dataclass(slots=True)
class VideoPhaseContactAnnotation:
    """GT timing labels for one validation video."""

    video: str
    fps: float
    boundaries: dict[str, BoundaryAnnotation] = field(default_factory=dict)
    analysis_confidence: float | None = None
    notes: str = ""

    def resolved_timestamp(self, name: str) -> float | None:
        """Return annotated timestamp, or frame_index / fps when timestamp omitted."""
        boundary = self.boundaries.get(name)
        if boundary is None:
            return None
        if boundary.timestamp is not None:
            return float(boundary.timestamp)
        if self.fps <= 0:
            return None
        return float(boundary.frame_index) / float(self.fps)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "video": self.video,
            "fps": float(self.fps),
            "boundaries": {
                name: b.to_dict() for name, b in sorted(self.boundaries.items())
            },
        }
        if self.analysis_confidence is not None:
            payload["analysis_confidence"] = float(self.analysis_confidence)
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(slots=True)
class PhaseContactAnnotationSet:
    """Collection of manually labeled validation videos."""

    videos: list[VideoPhaseContactAnnotation] = field(default_factory=list)
    annotation_version: str = PHASE_CONTACT_VALIDATION_ANNOTATION_VERSION
    notes: str = _DEFAULT_NOTES

    def by_video(self) -> dict[str, VideoPhaseContactAnnotation]:
        return {v.video: v for v in self.videos}

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_contact_validation_annotation_version": self.annotation_version,
            "video_count": len(self.videos),
            "notes": self.notes,
            "videos": [v.to_dict() for v in self.videos],
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def load_phase_contact_annotation_set(path: Path) -> PhaseContactAnnotationSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return phase_contact_annotation_set_from_dict(data)


def phase_contact_annotation_set_from_dict(
    data: dict[str, Any],
) -> PhaseContactAnnotationSet:
    version = str(
        data.get("phase_contact_validation_annotation_version")
        or data.get("annotation_version")
        or PHASE_CONTACT_VALIDATION_ANNOTATION_VERSION
    )
    # Support either {"videos": [...]} or a single-video object.
    raw_videos = data.get("videos")
    if raw_videos is None and "video" in data:
        raw_videos = [data]
    videos: list[VideoPhaseContactAnnotation] = []
    for raw in raw_videos or []:
        boundaries: dict[str, BoundaryAnnotation] = {}
        for name, vals in (raw.get("boundaries") or {}).items():
            if name not in BOUNDARY_NAMES:
                continue
            if not isinstance(vals, dict) or "frame_index" not in vals:
                continue
            ts = vals.get("timestamp")
            boundaries[name] = BoundaryAnnotation(
                frame_index=int(vals["frame_index"]),
                timestamp=float(ts) if ts is not None else None,
            )
        videos.append(
            VideoPhaseContactAnnotation(
                video=str(raw.get("video") or ""),
                fps=float(raw.get("fps") or 0.0),
                boundaries=boundaries,
                analysis_confidence=(
                    float(raw["analysis_confidence"])
                    if raw.get("analysis_confidence") is not None
                    else None
                ),
                notes=str(raw.get("notes") or ""),
            )
        )
    return PhaseContactAnnotationSet(
        videos=videos,
        annotation_version=version,
        notes=str(data.get("notes") or _DEFAULT_NOTES),
    )
