"""Manually annotated ground-truth frames for offline RTMPose validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.cv.skeleton import COCO_KEYPOINT_NAMES
from app.schemas.pose import Keypoint

POSE_VALIDATION_ANNOTATION_VERSION = "1.0.0"
_DEFAULT_NOTES = (
    "Ground-truth COCO-17 keypoints in normalized image coordinates [0, 1]. "
    "Used only by offline PoseValidator — not by /analyze."
)


@dataclass(slots=True)
class AnnotatedKeypoint:
    """Ground-truth joint in normalized image coordinates (no confidence)."""

    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y)}


@dataclass(slots=True)
class AnnotatedPoseFrame:
    """One manually labeled frame (subset of COCO-17 allowed)."""

    frame_index: int
    keypoints: dict[str, AnnotatedKeypoint] = field(default_factory=dict)
    # Optional explicit phase label; otherwise PoseValidator uses PhaseSequence.
    phase: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_index": int(self.frame_index),
            "keypoints": {name: kp.to_dict() for name, kp in self.keypoints.items()},
        }
        if self.phase is not None:
            payload["phase"] = self.phase
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(slots=True)
class PoseValidationAnnotationSet:
    """Small manually annotated validation set for one clip (or clip stem)."""

    video: str
    frames: list[AnnotatedPoseFrame] = field(default_factory=list)
    annotation_version: str = POSE_VALIDATION_ANNOTATION_VERSION
    notes: str = _DEFAULT_NOTES

    def frame_by_index(self) -> dict[int, AnnotatedPoseFrame]:
        return {f.frame_index: f for f in self.frames}

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose_validation_annotation_version": self.annotation_version,
            "video": self.video,
            "frame_count": len(self.frames),
            "notes": self.notes,
            "frames": [f.to_dict() for f in self.frames],
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def load_annotation_set(path: Path) -> PoseValidationAnnotationSet:
    """Load a versioned annotation JSON from disk."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return annotation_set_from_dict(data)


def annotation_set_from_dict(data: dict[str, Any]) -> PoseValidationAnnotationSet:
    version = str(
        data.get("pose_validation_annotation_version")
        or data.get("annotation_version")
        or POSE_VALIDATION_ANNOTATION_VERSION
    )
    frames: list[AnnotatedPoseFrame] = []
    for raw in data.get("frames") or []:
        kps: dict[str, AnnotatedKeypoint] = {}
        for name, vals in (raw.get("keypoints") or {}).items():
            if name not in COCO_KEYPOINT_NAMES:
                continue
            if not isinstance(vals, dict):
                continue
            if "x" not in vals or "y" not in vals:
                continue
            kps[name] = AnnotatedKeypoint(x=float(vals["x"]), y=float(vals["y"]))
        frames.append(
            AnnotatedPoseFrame(
                frame_index=int(raw["frame_index"]),
                keypoints=kps,
                phase=raw.get("phase"),
                notes=str(raw.get("notes") or ""),
            )
        )
    frames.sort(key=lambda f: f.frame_index)
    return PoseValidationAnnotationSet(
        video=str(data.get("video") or ""),
        frames=frames,
        annotation_version=version,
        notes=str(data.get("notes") or _DEFAULT_NOTES),
    )


def annotated_to_keypoint(ann: AnnotatedKeypoint, *, confidence: float = 1.0) -> Keypoint:
    """Convert GT annotation to a Keypoint (confidence unused for GT)."""
    return Keypoint(x=ann.x, y=ann.y, confidence=confidence)
