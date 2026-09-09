"""Versioned dataset export for coach validation and future model training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas.annotation import (
    ANNOTATION_SCHEMA_VERSION,
    CoachAnnotation,
    CoachAnnotationSet,
)

DATASET_EXPORT_VERSION = "1.0.0"


@dataclass(slots=True)
class DatasetExport:
    """Reproducible, label-ready snapshot of one analysis.

    Assembled from existing artifacts only — does not recompute pose, contact,
    or technique. Coach annotations start empty for independent labeling.
    """

    dataset_export_version: str
    analysis_id: str
    created_at: str
    stroke_type: str
    video_metadata: dict[str, Any]
    pose_metrics: dict[str, Any]
    phases: dict[str, Any]
    contact_event: dict[str, Any]
    technique_issues: list[dict[str, Any]]
    keyframes: list[dict[str, Any]]
    video_quality: dict[str, Any] | None = None
    artifact_refs: dict[str, str | None] = field(default_factory=dict)
    coach_annotations: CoachAnnotationSet = field(default_factory=CoachAnnotationSet)
    annotation_template: dict[str, Any] = field(default_factory=dict)
    notes: str = (
        "Label-ready dataset export. Analysis behavior is unchanged; this file "
        "mirrors persisted artifacts for coach validation and future training."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_export_version": self.dataset_export_version,
            "analysis_id": self.analysis_id,
            "created_at": self.created_at,
            "stroke_type": self.stroke_type,
            "video_metadata": self.video_metadata,
            "pose_metrics": self.pose_metrics,
            "phases": self.phases,
            "contact_event": self.contact_event,
            "technique_issues": self.technique_issues,
            "keyframes": self.keyframes,
            "video_quality": self.video_quality,
            "artifact_refs": self.artifact_refs,
            "coach_annotations": self.coach_annotations.to_dict(),
            "annotation_template": self.annotation_template,
            "notes": self.notes,
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def blank_annotation_template(analysis_id: str) -> dict[str, Any]:
    """Example CoachAnnotation JSON coaches can copy and fill."""
    template = CoachAnnotation.blank_template(
        analysis_id=analysis_id,
        coach_id="COACH_ID_HERE",
    )
    payload = template.to_dict()
    payload["annotated_at"] = utc_now_iso()
    payload["_instructions"] = (
        "Copy this object into a file named "
        f"{{analysis_id}}_annotation_{{coach_id}}.json. "
        f"annotation_version={ANNOTATION_SCHEMA_VERSION}. "
        "Rate preparation_quality, kinetic_chain_timing, contact_quality, "
        "follow_through_quality; add overall_issue_labels and free_text_notes. "
        "Each coach_id must be unique for this analysis_id."
    )
    return payload
