"""Shared artifact provenance envelope (analysis_id / snapshot_id / fingerprint)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Bump when the snapshot / provenance envelope shape changes.
ANALYSIS_SNAPSHOT_SCHEMA_VERSION = "1.0.0"

ARTIFACT_ROLE_FINAL = "final"
ARTIFACT_ROLE_INTERMEDIATE = "intermediate"

# Per-artifact payload schema versions (independent of snapshot envelope).
PHASES_ARTIFACT_SCHEMA_VERSION = "1.0.0"
CONTACT_ARTIFACT_SCHEMA_VERSION = "1.0.0"
STROKE_METRICS_ARTIFACT_SCHEMA_VERSION = "1.0.0"
TECHNIQUE_ARTIFACT_SCHEMA_VERSION = "1.0.0"
KEYFRAMES_ARTIFACT_SCHEMA_VERSION = "1.0.0"
COACHING_ARTIFACT_SCHEMA_VERSION = "1.0.0"
OVERLAY_META_ARTIFACT_SCHEMA_VERSION = "1.0.0"
QUALITY_ARTIFACT_SCHEMA_VERSION = "1.0.0"
ANGLES_ARTIFACT_SCHEMA_VERSION = "1.0.0"
MOTION_ARTIFACT_SCHEMA_VERSION = "1.0.0"
SMOOTHED_POSE_ARTIFACT_SCHEMA_VERSION = "1.0.0"
RAW_POSE_ARTIFACT_SCHEMA_VERSION = "1.0.0"

PROVENANCE_KEYS: tuple[str, ...] = (
    "analysis_id",
    "snapshot_id",
    "fingerprint",
    "snapshot_schema_version",
    "artifact_schema_version",
    "artifact_role",
)


class ProvenanceError(ValueError):
    """Raised when an artifact does not match the active AnalysisSnapshot."""


@dataclass(frozen=True, slots=True)
class AnalysisSnapshot:
    """Immutable provenance envelope for one finalized analysis."""

    analysis_id: str
    snapshot_id: str
    fingerprint: str
    snapshot_schema_version: str
    # Canonical payload that produced ``fingerprint`` (no paths / wall times).
    canonical: Mapping[str, Any]

    def provenance_dict(
        self,
        *,
        artifact_schema_version: str,
        artifact_role: str = ARTIFACT_ROLE_FINAL,
    ) -> dict[str, str]:
        return {
            "analysis_id": self.analysis_id,
            "snapshot_id": self.snapshot_id,
            "fingerprint": self.fingerprint,
            "snapshot_schema_version": self.snapshot_schema_version,
            "artifact_schema_version": artifact_schema_version,
            "artifact_role": artifact_role,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "snapshot_id": self.snapshot_id,
            "fingerprint": self.fingerprint,
            "snapshot_schema_version": self.snapshot_schema_version,
            "canonical": dict(self.canonical),
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def provenance_fields_from_object(obj: Any) -> dict[str, str]:
    """Return non-empty provenance attributes from an artifact object."""
    out: dict[str, str] = {}
    for key in PROVENANCE_KEYS:
        value = getattr(obj, key, "")
        if value:
            out[key] = str(value)
    return out


def stamp_payload(
    payload: Mapping[str, Any],
    snapshot: AnalysisSnapshot,
    *,
    artifact_schema_version: str,
    artifact_role: str = ARTIFACT_ROLE_FINAL,
) -> dict[str, Any]:
    """Return a copy of ``payload`` with provenance fields applied."""
    out = dict(payload)
    out.update(
        snapshot.provenance_dict(
            artifact_schema_version=artifact_schema_version,
            artifact_role=artifact_role,
        )
    )
    return out


def apply_provenance(
    obj: Any,
    snapshot: AnalysisSnapshot,
    *,
    artifact_schema_version: str,
    artifact_role: str = ARTIFACT_ROLE_FINAL,
) -> Any:
    """Set provenance attributes on a dataclass-like artifact object."""
    fields = snapshot.provenance_dict(
        artifact_schema_version=artifact_schema_version,
        artifact_role=artifact_role,
    )
    for key, value in fields.items():
        if not hasattr(obj, key):
            raise ProvenanceError(
                f"Artifact {type(obj).__name__} is missing provenance field {key!r}."
            )
        setattr(obj, key, value)
    return obj


def validate_artifact_provenance(
    payload: Mapping[str, Any],
    snapshot: AnalysisSnapshot,
    *,
    expect_role: str | None = ARTIFACT_ROLE_FINAL,
) -> None:
    """Fail clearly if payload is missing or stale relative to ``snapshot``."""
    missing = [
        k for k in ("analysis_id", "snapshot_id", "fingerprint") if k not in payload
    ]
    if missing:
        raise ProvenanceError(
            "Artifact is missing provenance fields "
            f"{missing}; cannot accept as belonging to active snapshot "
            f"{snapshot.snapshot_id}."
        )

    if payload.get("analysis_id") != snapshot.analysis_id:
        raise ProvenanceError(
            f"Stale artifact analysis_id={payload.get('analysis_id')!r} "
            f"does not match active analysis_id={snapshot.analysis_id!r}."
        )
    if payload.get("snapshot_id") != snapshot.snapshot_id:
        raise ProvenanceError(
            f"Stale artifact snapshot_id={payload.get('snapshot_id')!r} "
            f"does not match active snapshot_id={snapshot.snapshot_id!r}."
        )
    if payload.get("fingerprint") != snapshot.fingerprint:
        raise ProvenanceError(
            f"Stale artifact fingerprint={payload.get('fingerprint')!r} "
            f"does not match active fingerprint={snapshot.fingerprint!r}."
        )
    schema_ver = payload.get("snapshot_schema_version")
    if schema_ver is not None and schema_ver != snapshot.snapshot_schema_version:
        raise ProvenanceError(
            f"Artifact snapshot_schema_version={schema_ver!r} does not match "
            f"active {snapshot.snapshot_schema_version!r}."
        )
    if expect_role is not None:
        role = payload.get("artifact_role")
        if role is not None and role != expect_role:
            raise ProvenanceError(
                f"Artifact role={role!r} does not match expected {expect_role!r}."
            )


def validate_object_provenance(
    obj: Any,
    snapshot: AnalysisSnapshot,
    *,
    expect_role: str | None = ARTIFACT_ROLE_FINAL,
) -> None:
    """Validate provenance attributes already applied to an in-memory object."""
    validate_artifact_provenance(
        provenance_fields_from_object(obj),
        snapshot,
        expect_role=expect_role,
    )


def save_artifact_json(
    path: Path,
    payload: Mapping[str, Any],
    snapshot: AnalysisSnapshot,
    *,
    artifact_schema_version: str,
    artifact_role: str = ARTIFACT_ROLE_FINAL,
) -> Path:
    """Stamp, validate against active snapshot, then write JSON."""
    stamped = stamp_payload(
        payload,
        snapshot,
        artifact_schema_version=artifact_schema_version,
        artifact_role=artifact_role,
    )
    validate_artifact_provenance(
        stamped,
        snapshot,
        expect_role=artifact_role,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamped, indent=2), encoding="utf-8")
    return path


def overlay_metadata_dict(
    snapshot: AnalysisSnapshot,
    *,
    output_video_name: str,
    contact_frame_index: int,
    phase_contact_frame_index: int | None,
) -> dict[str, Any]:
    """Annotated-video sidecar metadata (no pixel data)."""
    return stamp_payload(
        {
            "video": output_video_name,
            "contact_frame_index": contact_frame_index,
            "phase_contact_frame_index": phase_contact_frame_index,
            "notes": (
                "Annotated overlay video rendered from FinalAnalysisState; "
                "provenance ties the mp4 to this snapshot."
            ),
        },
        snapshot,
        artifact_schema_version=OVERLAY_META_ARTIFACT_SCHEMA_VERSION,
        artifact_role=ARTIFACT_ROLE_FINAL,
    )
