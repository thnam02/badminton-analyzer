"""Typed coach annotation schemas for dataset labeling (no ML training yet)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

ANNOTATION_SCHEMA_VERSION = "1.0.0"


class QualityRating(str, Enum):
    """Ordered qualitative score for stroke facets."""

    NOT_RATED = "NOT_RATED"
    POOR = "POOR"
    FAIR = "FAIR"
    GOOD = "GOOD"
    EXCELLENT = "EXCELLENT"


# Optional numeric mapping for training later (not used by analysis).
QUALITY_RATING_TO_SCORE: dict[str, int | None] = {
    QualityRating.NOT_RATED.value: None,
    QualityRating.POOR.value: 1,
    QualityRating.FAIR.value: 2,
    QualityRating.GOOD.value: 3,
    QualityRating.EXCELLENT.value: 4,
}


@dataclass(slots=True)
class QualityScore:
    """Coach rating for one stroke facet."""

    rating: str = QualityRating.NOT_RATED.value
    score: int | None = None  # optional 1–4; defaults from rating when omitted
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        score = self.score
        if score is None:
            score = QUALITY_RATING_TO_SCORE.get(self.rating)
        return {
            "rating": self.rating,
            "score": score,
            "notes": self.notes,
        }

    @classmethod
    def blank(cls) -> QualityScore:
        return cls()


@dataclass(slots=True)
class CoachAnnotation:
    """One coach's independent labels for a single analyzed stroke.

    Multiple coaches label the same ``analysis_id`` in separate files; do not
    overwrite each other. See docs/dataset-annotations.md.
    """

    annotation_version: str = ANNOTATION_SCHEMA_VERSION
    analysis_id: str = ""
    coach_id: str = ""
    annotated_at: str | None = None
    preparation_quality: QualityScore = field(default_factory=QualityScore.blank)
    kinetic_chain_timing: QualityScore = field(default_factory=QualityScore.blank)
    contact_quality: QualityScore = field(default_factory=QualityScore.blank)
    follow_through_quality: QualityScore = field(default_factory=QualityScore.blank)
    overall_issue_labels: list[str] = field(default_factory=list)
    free_text_notes: str = ""
    # Optional contact validation against the system ContactEvent.
    agrees_with_system_contact: bool | None = None
    corrected_contact_frame_index: int | None = None
    corrected_contact_timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_version": self.annotation_version,
            "analysis_id": self.analysis_id,
            "coach_id": self.coach_id,
            "annotated_at": self.annotated_at,
            "preparation_quality": self.preparation_quality.to_dict(),
            "kinetic_chain_timing": self.kinetic_chain_timing.to_dict(),
            "contact_quality": self.contact_quality.to_dict(),
            "follow_through_quality": self.follow_through_quality.to_dict(),
            "overall_issue_labels": list(self.overall_issue_labels),
            "free_text_notes": self.free_text_notes,
            "agrees_with_system_contact": self.agrees_with_system_contact,
            "corrected_contact_frame_index": self.corrected_contact_frame_index,
            "corrected_contact_timestamp": self.corrected_contact_timestamp,
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def blank_template(cls, *, analysis_id: str, coach_id: str = "") -> CoachAnnotation:
        return cls(
            analysis_id=analysis_id,
            coach_id=coach_id,
            annotated_at=None,
            preparation_quality=QualityScore.blank(),
            kinetic_chain_timing=QualityScore.blank(),
            contact_quality=QualityScore.blank(),
            follow_through_quality=QualityScore.blank(),
            overall_issue_labels=[],
            free_text_notes="",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoachAnnotation:
        def _qs(raw: dict[str, Any] | None) -> QualityScore:
            raw = raw or {}
            return QualityScore(
                rating=str(raw.get("rating", QualityRating.NOT_RATED.value)),
                score=raw.get("score"),
                notes=str(raw.get("notes", "")),
            )

        return cls(
            annotation_version=str(
                data.get("annotation_version", ANNOTATION_SCHEMA_VERSION)
            ),
            analysis_id=str(data.get("analysis_id", "")),
            coach_id=str(data.get("coach_id", "")),
            annotated_at=data.get("annotated_at"),
            preparation_quality=_qs(data.get("preparation_quality")),
            kinetic_chain_timing=_qs(data.get("kinetic_chain_timing")),
            contact_quality=_qs(data.get("contact_quality")),
            follow_through_quality=_qs(data.get("follow_through_quality")),
            overall_issue_labels=[
                str(x) for x in (data.get("overall_issue_labels") or [])
            ],
            free_text_notes=str(data.get("free_text_notes", "")),
            agrees_with_system_contact=data.get("agrees_with_system_contact"),
            corrected_contact_frame_index=data.get("corrected_contact_frame_index"),
            corrected_contact_timestamp=data.get("corrected_contact_timestamp"),
        )


@dataclass(slots=True)
class CoachAnnotationSet:
    """Container for zero-or-more independent coach labels on one stroke."""

    schema: str = "coach_annotation_set_v1"
    annotation_version: str = ANNOTATION_SCHEMA_VERSION
    analysis_id: str = ""
    annotations: list[CoachAnnotation] = field(default_factory=list)
    notes: str = (
        "Placeholders for independent coach labels. Each coach submits a separate "
        "CoachAnnotation JSON keyed by coach_id; do not merge over another coach."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "annotation_version": self.annotation_version,
            "analysis_id": self.analysis_id,
            "annotation_count": len(self.annotations),
            "annotations": [a.to_dict() for a in self.annotations],
            "notes": self.notes,
        }
