"""Build AnalysisSnapshot + deterministic fingerprint from FinalAnalysisState."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from app.schemas.final_analysis import FinalAnalysisState
from app.schemas.provenance import (
    ANALYSIS_SNAPSHOT_SCHEMA_VERSION,
    AnalysisSnapshot,
    ProvenanceError,
)


def build_analysis_snapshot(
    state: FinalAnalysisState,
    *,
    analysis_id: str,
) -> AnalysisSnapshot:
    """Derive a deterministic snapshot + fingerprint from finalized state."""
    if not analysis_id:
        raise ProvenanceError("analysis_id is required to build AnalysisSnapshot.")
    canonical = build_canonical_fingerprint_payload(state)
    fingerprint = fingerprint_canonical_payload(canonical)
    snapshot_id = f"snap_{fingerprint[:16]}"
    return AnalysisSnapshot(
        analysis_id=analysis_id,
        snapshot_id=snapshot_id,
        fingerprint=fingerprint,
        snapshot_schema_version=ANALYSIS_SNAPSHOT_SCHEMA_VERSION,
        canonical=canonical,
    )


def build_canonical_fingerprint_payload(state: FinalAnalysisState) -> dict[str, Any]:
    """Stable subset of finalized analysis used for hashing.

    Excludes temporary file paths, wall-clock timestamps, and free-form notes.
    """
    contact = state.contact
    phases = state.phases
    contact_f = int(contact.frame_index)

    angle_at_contact: dict[str, float | None] = {}
    for frame in state.angles.frames:
        if frame.frame_index == contact_f:
            angle_at_contact = {
                "right_elbow": _round_opt(frame.right_elbow),
                "right_knee": _round_opt(frame.right_knee),
                "right_shoulder": _round_opt(frame.right_shoulder),
            }
            break

    motion_at_contact: dict[str, float | None] = {}
    for frame in state.motion.frames:
        if frame.frame_index == contact_f:
            motion_at_contact = {
                "right_wrist_speed": _round_opt(frame.right_wrist_speed),
                "right_elbow_angular_velocity": _round_opt(
                    frame.right_elbow_angular_velocity
                ),
                "right_knee_angular_velocity": _round_opt(
                    frame.right_knee_angular_velocity
                ),
            }
            break

    peaks: dict[str, Any] = {}
    for name, peak in sorted((state.motion.peaks or {}).items()):
        peaks[name] = {
            "value": _round_opt(peak.value),
            "frame_index": peak.frame_index,
        }

    quality = state.video_quality
    metrics = quality.metrics.to_dict() if quality.metrics is not None else {}
    quality_metrics = {
        key: _round_opt(value) if isinstance(value, float) else value
        for key, value in sorted(metrics.items())
    }

    segments = [
        {
            "phase": seg.phase.value,
            "start_frame_index": int(seg.start_frame_index),
            "end_frame_index": int(seg.end_frame_index),
            "confidence": _round_opt(seg.confidence),
        }
        for seg in phases.segments
    ]

    return {
        "analysis_snapshot_schema_version": ANALYSIS_SNAPSHOT_SCHEMA_VERSION,
        "stroke_metadata": {
            "video": state.video,
            "video_fps": _round_opt(state.video_fps),
            "video_width": int(state.video_width),
            "video_height": int(state.video_height),
            "frame_count": len(state.frame_indices),
            "frame_index_min": state.frame_indices[0] if state.frame_indices else None,
            "frame_index_max": state.frame_indices[-1] if state.frame_indices else None,
        },
        "contact": {
            "contact_type": contact.contact_type,
            "frame_index": int(contact.frame_index),
            "confidence": _round_opt(contact.confidence),
            "kinematic_frame_index": contact.kinematic_frame_index,
        },
        "phases": {
            "estimated_contact_frame_index": phases.estimated_contact_frame_index,
            "confidence": _round_opt(phases.confidence),
            "segments": segments,
        },
        "motion_summary": {
            "frame_count": state.motion.frame_count,
            "peaks": peaks,
            "at_contact": motion_at_contact,
        },
        "angle_summary": {
            "frame_count": state.angles.frame_count,
            "at_contact": angle_at_contact,
        },
        "quality": {
            "usable": bool(quality.usable),
            "analysis_confidence": _round_opt(quality.analysis_confidence),
            "metrics": quality_metrics,
            "warnings": list(quality.warnings),
        },
    }


def fingerprint_canonical_payload(canonical: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of canonical JSON (sorted keys, compact separators)."""
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round_opt(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)
