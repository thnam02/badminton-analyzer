"""Manual coach annotations for offline technique-issue validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.validation.technique_issues import (
    COACH_LABELS,
    LABEL_ABSENT,
    LABEL_PRESENT,
    LABEL_UNCERTAIN,
    SUPPORTED_TECHNIQUE_ISSUE_CODES,
)

TECHNIQUE_VALIDATION_ANNOTATION_VERSION = "1.0.0"
_DEFAULT_NOTES = (
    "Coach ground-truth labels for technique issues (present / absent / uncertain). "
    "Used only by offline TechniqueIssueValidator — not by /analyze."
)


@dataclass(slots=True)
class CoachIssueLabel:
    """One coach label for a supported technique issue on a stroke."""

    label: str
    coach_confidence: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        normalized = str(self.label).strip().lower()
        if normalized not in COACH_LABELS:
            raise ValueError(
                f"Invalid coach label {self.label!r}; expected one of {COACH_LABELS}"
            )
        self.label = normalized

    @property
    def is_certain(self) -> bool:
        return self.label in (LABEL_PRESENT, LABEL_ABSENT)

    @property
    def is_present(self) -> bool:
        return self.label == LABEL_PRESENT

    @property
    def is_uncertain(self) -> bool:
        return self.label == LABEL_UNCERTAIN

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"label": self.label}
        if self.coach_confidence is not None:
            payload["coach_confidence"] = float(self.coach_confidence)
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(slots=True)
class StrokeTechniqueAnnotation:
    """Coach labels for one validation stroke / clip."""

    stroke_id: str
    video: str = ""
    issues: dict[str, CoachIssueLabel] = field(default_factory=dict)
    analysis_confidence: float | None = None
    camera_quality: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stroke_id": self.stroke_id,
            "video": self.video or self.stroke_id,
            "issues": {
                code: label.to_dict() for code, label in sorted(self.issues.items())
            },
        }
        if self.analysis_confidence is not None:
            payload["analysis_confidence"] = float(self.analysis_confidence)
        if self.camera_quality is not None:
            payload["camera_quality"] = str(self.camera_quality)
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(slots=True)
class TechniqueValidationAnnotationSet:
    """Collection of coach-labelled validation strokes."""

    strokes: list[StrokeTechniqueAnnotation] = field(default_factory=list)
    annotation_version: str = TECHNIQUE_VALIDATION_ANNOTATION_VERSION
    notes: str = _DEFAULT_NOTES

    def by_stroke_id(self) -> dict[str, StrokeTechniqueAnnotation]:
        return {s.stroke_id: s for s in self.strokes}

    def to_dict(self) -> dict[str, Any]:
        return {
            "technique_validation_annotation_version": self.annotation_version,
            "stroke_count": len(self.strokes),
            "notes": self.notes,
            "strokes": [s.to_dict() for s in self.strokes],
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def load_technique_annotation_set(path: Path) -> TechniqueValidationAnnotationSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return technique_annotation_set_from_dict(data)


def technique_annotation_set_from_dict(
    data: dict[str, Any],
) -> TechniqueValidationAnnotationSet:
    version = str(
        data.get("technique_validation_annotation_version")
        or data.get("annotation_version")
        or TECHNIQUE_VALIDATION_ANNOTATION_VERSION
    )
    raw_strokes = data.get("strokes")
    if raw_strokes is None and "stroke_id" in data:
        raw_strokes = [data]
    strokes: list[StrokeTechniqueAnnotation] = []
    for raw in raw_strokes or []:
        issues: dict[str, CoachIssueLabel] = {}
        for code, vals in (raw.get("issues") or {}).items():
            if code not in SUPPORTED_TECHNIQUE_ISSUE_CODES:
                continue
            if isinstance(vals, str):
                issues[code] = CoachIssueLabel(label=vals)
                continue
            if not isinstance(vals, dict) or "label" not in vals:
                continue
            conf = vals.get("coach_confidence")
            issues[code] = CoachIssueLabel(
                label=str(vals["label"]),
                coach_confidence=float(conf) if conf is not None else None,
                notes=str(vals.get("notes") or ""),
            )
        stroke_id = str(raw.get("stroke_id") or raw.get("video") or "")
        strokes.append(
            StrokeTechniqueAnnotation(
                stroke_id=stroke_id,
                video=str(raw.get("video") or stroke_id),
                issues=issues,
                analysis_confidence=(
                    float(raw["analysis_confidence"])
                    if raw.get("analysis_confidence") is not None
                    else None
                ),
                camera_quality=(
                    str(raw["camera_quality"])
                    if raw.get("camera_quality") is not None
                    else None
                ),
                notes=str(raw.get("notes") or ""),
            )
        )
    return TechniqueValidationAnnotationSet(
        strokes=strokes,
        annotation_version=version,
        notes=str(data.get("notes") or _DEFAULT_NOTES),
    )


# Re-export label constants for callers that import annotations only.
__all__ = [
    "TECHNIQUE_VALIDATION_ANNOTATION_VERSION",
    "CoachIssueLabel",
    "StrokeTechniqueAnnotation",
    "TechniqueValidationAnnotationSet",
    "load_technique_annotation_set",
    "technique_annotation_set_from_dict",
    "LABEL_ABSENT",
    "LABEL_PRESENT",
    "LABEL_UNCERTAIN",
]
