"""Manually annotated frames for offline joint-angle validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.cv.skeleton import COCO_KEYPOINT_NAMES
from app.validation.angle_joints import VALIDATED_ANGLE_NAMES
from app.validation.annotations import AnnotatedKeypoint

ANGLE_VALIDATION_ANNOTATION_VERSION = "1.0.0"
_DEFAULT_NOTES = (
    "Ground-truth joint angles (degrees) and/or COCO-17 keypoints in normalized "
    "coordinates. Used only by offline AngleValidator — not by /analyze."
)


@dataclass(slots=True)
class AnnotatedAngleFrame:
    """One labeled frame: direct GT angles and/or keypoints to derive them."""

    frame_index: int
    # Direct ground-truth angles in degrees (preferred when present).
    angles: dict[str, float] = field(default_factory=dict)
    # Optional keypoints — used to derive angles via the production formula.
    keypoints: dict[str, AnnotatedKeypoint] = field(default_factory=dict)
    phase: str | None = None
    # Optional frame-level pose confidence for confidence-binned error reports.
    pose_confidence: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"frame_index": int(self.frame_index)}
        if self.angles:
            payload["angles"] = {k: float(v) for k, v in self.angles.items()}
        if self.keypoints:
            payload["keypoints"] = {
                name: kp.to_dict() for name, kp in self.keypoints.items()
            }
        if self.phase is not None:
            payload["phase"] = self.phase
        if self.pose_confidence is not None:
            payload["pose_confidence"] = float(self.pose_confidence)
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(slots=True)
class AngleValidationAnnotationSet:
    """Small manually annotated set for angle validation on one clip."""

    video: str
    frames: list[AnnotatedAngleFrame] = field(default_factory=list)
    annotation_version: str = ANGLE_VALIDATION_ANNOTATION_VERSION
    notes: str = _DEFAULT_NOTES

    def frame_by_index(self) -> dict[int, AnnotatedAngleFrame]:
        return {f.frame_index: f for f in self.frames}

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle_validation_annotation_version": self.annotation_version,
            "video": self.video,
            "frame_count": len(self.frames),
            "notes": self.notes,
            "frames": [f.to_dict() for f in self.frames],
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def load_angle_annotation_set(path: Path) -> AngleValidationAnnotationSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return angle_annotation_set_from_dict(data)


def angle_annotation_set_from_dict(data: dict[str, Any]) -> AngleValidationAnnotationSet:
    version = str(
        data.get("angle_validation_annotation_version")
        or data.get("annotation_version")
        or ANGLE_VALIDATION_ANNOTATION_VERSION
    )
    frames: list[AnnotatedAngleFrame] = []
    for raw in data.get("frames") or []:
        angles: dict[str, float] = {}
        for name, value in (raw.get("angles") or {}).items():
            if name not in VALIDATED_ANGLE_NAMES:
                continue
            if value is None:
                continue
            angles[name] = float(value)

        kps: dict[str, AnnotatedKeypoint] = {}
        for name, vals in (raw.get("keypoints") or {}).items():
            if name not in COCO_KEYPOINT_NAMES:
                continue
            if not isinstance(vals, dict) or "x" not in vals or "y" not in vals:
                continue
            kps[name] = AnnotatedKeypoint(x=float(vals["x"]), y=float(vals["y"]))

        pose_conf = raw.get("pose_confidence")
        frames.append(
            AnnotatedAngleFrame(
                frame_index=int(raw["frame_index"]),
                angles=angles,
                keypoints=kps,
                phase=raw.get("phase"),
                pose_confidence=(
                    float(pose_conf) if pose_conf is not None else None
                ),
                notes=str(raw.get("notes") or ""),
            )
        )
    frames.sort(key=lambda f: f.frame_index)
    return AngleValidationAnnotationSet(
        video=str(data.get("video") or ""),
        frames=frames,
        annotation_version=version,
        notes=str(data.get("notes") or _DEFAULT_NOTES),
    )
