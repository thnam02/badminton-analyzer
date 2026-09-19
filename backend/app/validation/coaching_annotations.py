"""Optional coach-review annotations for offline coaching validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COACHING_VALIDATION_ANNOTATION_VERSION = "1.0.0"

COACH_RATING_DIMENSIONS: tuple[str, ...] = (
    "priority_agreement",
    "technical_correctness",
    "actionability",
    "clarity",
    "drill_relevance",
)

_DEFAULT_NOTES = (
    "Optional human coach review of generated CoachingReports (1–5 ratings). "
    "Used only by offline CoachingValidator — not by /analyze."
)


def _clamp_rating(value: int | float) -> int:
    rating = int(value)
    if rating < 1 or rating > 5:
        raise ValueError(f"Coach rating must be in 1–5, got {value!r}")
    return rating


@dataclass(slots=True)
class CoachReviewAnnotation:
    """1–5 coach ratings for one generated coaching analysis."""

    analysis_id: str
    video: str = ""
    priority_agreement: int | None = None
    technical_correctness: int | None = None
    actionability: int | None = None
    clarity: int | None = None
    drill_relevance: int | None = None
    # Optional expected top issue codes for automatic priority agreement.
    expected_priority_issue_codes: list[str] = field(default_factory=list)
    analysis_confidence: float | None = None
    comments: str = ""

    def __post_init__(self) -> None:
        for name in COACH_RATING_DIMENSIONS:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _clamp_rating(value))

    def ratings(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name in COACH_RATING_DIMENSIONS:
            value = getattr(self, name)
            if value is not None:
                out[name] = int(value)
        return out

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "analysis_id": self.analysis_id,
            "video": self.video or self.analysis_id,
        }
        for name in COACH_RATING_DIMENSIONS:
            value = getattr(self, name)
            if value is not None:
                payload[name] = int(value)
        if self.expected_priority_issue_codes:
            payload["expected_priority_issue_codes"] = list(
                self.expected_priority_issue_codes
            )
        if self.analysis_confidence is not None:
            payload["analysis_confidence"] = float(self.analysis_confidence)
        if self.comments:
            payload["comments"] = self.comments
        return payload


@dataclass(slots=True)
class CoachingValidationAnnotationSet:
    reviews: list[CoachReviewAnnotation] = field(default_factory=list)
    annotation_version: str = COACHING_VALIDATION_ANNOTATION_VERSION
    notes: str = _DEFAULT_NOTES

    def by_analysis_id(self) -> dict[str, CoachReviewAnnotation]:
        return {r.analysis_id: r for r in self.reviews}

    def to_dict(self) -> dict[str, Any]:
        return {
            "coaching_validation_annotation_version": self.annotation_version,
            "review_count": len(self.reviews),
            "notes": self.notes,
            "reviews": [r.to_dict() for r in self.reviews],
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def load_coaching_annotation_set(path: Path) -> CoachingValidationAnnotationSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return coaching_annotation_set_from_dict(data)


def coaching_annotation_set_from_dict(
    data: dict[str, Any],
) -> CoachingValidationAnnotationSet:
    version = str(
        data.get("coaching_validation_annotation_version")
        or data.get("annotation_version")
        or COACHING_VALIDATION_ANNOTATION_VERSION
    )
    raw_reviews = data.get("reviews")
    if raw_reviews is None and "analysis_id" in data:
        raw_reviews = [data]
    reviews: list[CoachReviewAnnotation] = []
    for raw in raw_reviews or []:
        ratings = {
            name: int(raw[name]) if raw.get(name) is not None else None
            for name in COACH_RATING_DIMENSIONS
        }
        expected = raw.get("expected_priority_issue_codes") or []
        reviews.append(
            CoachReviewAnnotation(
                analysis_id=str(raw.get("analysis_id") or raw.get("video") or ""),
                video=str(raw.get("video") or raw.get("analysis_id") or ""),
                priority_agreement=ratings["priority_agreement"],
                technical_correctness=ratings["technical_correctness"],
                actionability=ratings["actionability"],
                clarity=ratings["clarity"],
                drill_relevance=ratings["drill_relevance"],
                expected_priority_issue_codes=[str(c) for c in expected],
                analysis_confidence=(
                    float(raw["analysis_confidence"])
                    if raw.get("analysis_confidence") is not None
                    else None
                ),
                comments=str(raw.get("comments") or ""),
            )
        )
    return CoachingValidationAnnotationSet(
        reviews=reviews,
        annotation_version=version,
        notes=str(data.get("notes") or _DEFAULT_NOTES),
    )
