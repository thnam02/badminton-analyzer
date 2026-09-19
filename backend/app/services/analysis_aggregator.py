"""Aggregate finalized analysis artifacts into a product-facing DTO (Phase D).

Reads existing outputs only — never recomputes biomechanics or coaching.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.analysis_index import (
    AnalysisIndexEntry,
    load_index,
    upsert_index_entry,
)
from app.services.video_service import (
    _artifact_base_stem,
    analysis_snapshot_json_path_for,
    coaching_json_path_for,
    contact_json_path_for,
    evidence_json_path_for,
    keyframes_json_path_for,
    overlay_meta_json_path_for,
    phases_json_path_for,
    stroke_metrics_json_path_for,
    technique_json_path_for,
    video_quality_json_path_for,
)

# User-facing phase labels (backend ESTIMATED_CONTACT → Estimated contact).
PHASE_LABELS = {
    "PREPARATION": "Preparation",
    "BACKSWING": "Backswing",
    "ACCELERATION": "Acceleration",
    "ESTIMATED_CONTACT": "Estimated contact",
    "CONTACT": "Contact",
    "FOLLOW_THROUGH": "Follow-through",
    "RECOVERY": "Recovery",
}

ISSUE_TITLES = {
    "INSUFFICIENT_ELBOW_EXTENSION": "Limited elbow extension",
    "LOW_KNEE_CONTRIBUTION": "Limited knee contribution",
    "PREPARATION_KNEE_OUT_OF_RANGE": "Preparation knee position",
    "POOR_ARM_ACCELERATION_TIMING": "Arm acceleration timing",
    "LOW_CONTACT_POSTURE": "Contact height",
    "WEAK_FOLLOW_THROUGH": "Follow-through",
    "LIMITED_CLEAR_PREPARATION": "Limited clear preparation",
    "INSUFFICIENT_ARM_EXTENSION": "Insufficient arm extension",
    "POOR_PROXIMAL_DISTAL_TIMING": "Proximal-to-distal timing",
    "RESTRICTED_FOLLOW_THROUGH": "Restricted follow-through",
    "SLOW_RECOVERY": "Slow recovery",
    "INSUFFICIENT_EVIDENCE": "Insufficient evidence",
}


def list_analyses(*, output_dir: Path | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent analysis summaries (manifest + disk discovery)."""
    root = Path(output_dir) if output_dir is not None else Path(settings.output_dir)
    discovered = _discover_analyses(root)
    # Merge manifest metadata over discovery.
    by_id = {e["analysis_id"]: e for e in discovered}
    for entry in load_index(root):
        base = by_id.get(entry.analysis_id, {})
        by_id[entry.analysis_id] = {
            **base,
            **{k: v for k, v in entry.to_dict().items() if v not in (None, "", "UNKNOWN")},
            "analysis_id": entry.analysis_id,
        }
    items = sorted(
        by_id.values(),
        key=lambda x: str(x.get("created_at") or ""),
        reverse=True,
    )
    return items[: max(1, int(limit))]


def get_analysis_result(
    analysis_id: str, *, output_dir: Path | None = None
) -> dict[str, Any]:
    """Load and normalize all product-facing fields for one analysis."""
    root = Path(output_dir) if output_dir is not None else Path(settings.output_dir)
    video_path = root / f"{analysis_id}_pose.mp4"
    if not video_path.is_file():
        # Accept stem without requiring video for partial fixtures.
        if not any(root.glob(f"{analysis_id}_pose*")):
            raise FileNotFoundError(f"Analysis '{analysis_id}' not found")

    phases = _read_json(phases_json_path_for(video_path))
    metrics = _read_json(stroke_metrics_json_path_for(video_path))
    technique = _read_json(technique_json_path_for(video_path))
    quality = _read_json(video_quality_json_path_for(video_path))
    contact = _read_json(contact_json_path_for(video_path))
    evidence = _read_json(evidence_json_path_for(video_path))
    coaching = _read_json(coaching_json_path_for(video_path))
    keyframes = _read_json(keyframes_json_path_for(video_path))
    snapshot = _read_json(analysis_snapshot_json_path_for(video_path))
    overlay_meta = _read_json(overlay_meta_json_path_for(video_path))

    analysis_status = "COMPLETE" if (phases or metrics or technique) else "PARTIAL"
    if not video_path.is_file():
        analysis_status = "PARTIAL"

    coaching_status = _coaching_status(coaching)
    mesh_status = _mesh_status(root, analysis_id)

    contact_norm = _normalize_contact(contact, phases, evidence, metrics)
    phases_norm = _normalize_phases(phases, contact_norm)
    issues_norm = _normalize_issues(technique, contact_norm, phases_norm)
    confidence = _normalize_confidence(quality, phases, contact_norm, technique, evidence)
    coaching_norm = _normalize_coaching(coaching, issues_norm)
    metrics_norm = _normalize_metrics(metrics)
    references = _reference_comparisons(issues_norm, technique)

    handedness = (
        (evidence or {}).get("handedness")
        or (technique or {}).get("handedness")
        or None
    )
    stroke_type = (evidence or {}).get("stroke_type") or "SMASH"
    ref_profile = (
        (technique or {}).get("reference_profile_id")
        or (issues_norm[0]["reference_profile_id"] if issues_norm else "")
        or ""
    )

    main_issue = None
    for issue in issues_norm:
        if issue.get("status") != "INSUFFICIENT_EVIDENCE" and issue.get("is_finding"):
            main_issue = issue.get("code")
            break

    created_at = _file_created_at(video_path if video_path.is_file() else root)

    result = {
        "analysis_id": analysis_id,
        "created_at": created_at,
        "stroke_type": _stroke_label(stroke_type),
        "stroke_type_raw": stroke_type,
        "handedness": _hand_label(handedness),
        "handedness_raw": handedness,
        "camera_view": None,
        "reference_profile_id": ref_profile,
        "snapshot_id": (snapshot or {}).get("snapshot_id")
        or (technique or {}).get("snapshot_id")
        or "",
        "analysis_status": analysis_status,
        "coaching_status": coaching_status,
        "mesh_status": mesh_status,
        "video": {
            "pose_video_url": f"/outputs/{video_path.name}" if video_path.is_file() else None,
            "available": video_path.is_file(),
            "overlay_modes": _overlay_modes(overlay_meta),
        },
        "confidence": confidence,
        "phases": phases_norm,
        "contact": contact_norm,
        "issues": issues_norm,
        "insufficient_evidence": [
            i for i in issues_norm if i.get("status") == "INSUFFICIENT_EVIDENCE"
        ],
        "findings": [i for i in issues_norm if i.get("is_finding")],
        "metrics": metrics_norm,
        "reference_comparisons": references,
        "coaching": coaching_norm,
        "keyframes": _normalize_keyframes(keyframes, analysis_id),
        "limitations": _limitations(confidence, contact_norm, quality),
        "artifacts": {
            "phases": phases is not None,
            "metrics": metrics is not None,
            "technique": technique is not None,
            "quality": quality is not None,
            "contact": contact is not None,
            "evidence": evidence is not None,
            "coaching": coaching is not None,
            "keyframes": keyframes is not None,
        },
        "main_issue": main_issue,
        "analysis_confidence": confidence.get("overall_score"),
    }
    return result


def register_completed_analysis(
    analysis_id: str,
    *,
    output_dir: Path | None = None,
    mesh_status: str | None = None,
) -> AnalysisIndexEntry:
    """Write/update history index after /analyze completes."""
    result = get_analysis_result(analysis_id, output_dir=output_dir)
    entry = AnalysisIndexEntry(
        analysis_id=analysis_id,
        created_at=str(result.get("created_at") or _utc_now()),
        stroke_type=str(result.get("stroke_type_raw") or "forehand_smash").lower(),
        handedness=result.get("handedness_raw"),
        camera_view=result.get("camera_view"),
        analysis_confidence=result.get("analysis_confidence"),
        main_issue=result.get("main_issue"),
        pose_video_url=str((result.get("video") or {}).get("pose_video_url") or ""),
        snapshot_id=str(result.get("snapshot_id") or ""),
        reference_profile_id=str(result.get("reference_profile_id") or ""),
        analysis_status=str(result.get("analysis_status") or "COMPLETE"),
        coaching_status=str(result.get("coaching_status") or "UNAVAILABLE"),
        mesh_status=mesh_status or str(result.get("mesh_status") or "NONE"),
    )
    return upsert_index_entry(entry, output_dir=output_dir)


def compare_analyses(
    left_id: str,
    right_id: str,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    left = get_analysis_result(left_id, output_dir=output_dir)
    right = get_analysis_result(right_id, output_dir=output_dir)
    compatible, reason = _compatibility(left, right)
    rows = _comparison_rows(left, right) if compatible else []
    return {
        "compatible": compatible,
        "compatibility_reason": reason,
        "left": _summary_from_result(left),
        "right": _summary_from_result(right),
        "rows": rows,
    }


def _discover_analyses(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    found: dict[str, dict[str, Any]] = {}
    for video in root.glob("*_pose.mp4"):
        aid = _artifact_base_stem(video)
        if not aid:
            continue
        try:
            partial = get_analysis_result(aid, output_dir=root)
        except FileNotFoundError:
            continue
        found[aid] = _summary_from_result(partial)
    return list(found.values())


def _summary_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "analysis_id": result["analysis_id"],
        "created_at": result.get("created_at"),
        "stroke_type": result.get("stroke_type"),
        "stroke_type_raw": result.get("stroke_type_raw"),
        "handedness": result.get("handedness"),
        "handedness_raw": result.get("handedness_raw"),
        "analysis_confidence": result.get("analysis_confidence"),
        "main_issue": result.get("main_issue"),
        "pose_video_url": (result.get("video") or {}).get("pose_video_url"),
        "snapshot_id": result.get("snapshot_id"),
        "reference_profile_id": result.get("reference_profile_id"),
        "analysis_status": result.get("analysis_status"),
        "coaching_status": result.get("coaching_status"),
        "mesh_status": result.get("mesh_status"),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _normalize_phases(
    phases: dict[str, Any] | None, contact: dict[str, Any]
) -> list[dict[str, Any]]:
    if not phases:
        return []
    segments = phases.get("segments") or []
    out: list[dict[str, Any]] = []
    for seg in segments:
        phase = str(seg.get("phase") or "")
        is_contact = phase in {"ESTIMATED_CONTACT", "CONTACT"}
        out.append(
            {
                "id": phase,
                "label": PHASE_LABELS.get(phase, phase.replace("_", " ").title()),
                "start_timestamp": float(seg.get("start_timestamp") or 0.0),
                "end_timestamp": float(seg.get("end_timestamp") or 0.0),
                "start_frame_index": seg.get("start_frame_index"),
                "end_frame_index": seg.get("end_frame_index"),
                "confidence": float(seg.get("confidence") or phases.get("confidence") or 0.0),
                "is_contact_event": is_contact,
                "seek_timestamp": (
                    float(contact.get("timestamp"))
                    if is_contact and contact.get("timestamp") is not None
                    else float(seg.get("start_timestamp") or 0.0)
                ),
            }
        )
    return out


def _normalize_contact(
    contact: dict[str, Any] | None,
    phases: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    src = contact or (evidence or {}).get("contact") or {}
    if isinstance(src, dict) and "contact" in src and isinstance(src["contact"], dict):
        src = src["contact"]
    timestamp = src.get("timestamp")
    if timestamp is None and phases:
        timestamp = phases.get("estimated_contact_timestamp")
    if timestamp is None and metrics:
        timestamp = metrics.get("estimated_contact_timestamp")
    frame = src.get("frame_index")
    if frame is None and phases:
        frame = phases.get("estimated_contact_frame_index")
    if frame is None and metrics:
        frame = metrics.get("estimated_contact_frame_index")
    contact_type = str(src.get("contact_type") or "KINEMATIC_ESTIMATE")
    tracked = contact_type.upper() in {"TRACKED", "TRACKED_CONTACT"}
    return {
        "timestamp": float(timestamp) if timestamp is not None else None,
        "frame_index": int(frame) if frame is not None else None,
        "confidence": float(src.get("confidence") or 0.0),
        "contact_type": contact_type,
        "label": "Detected contact" if tracked else "Estimated contact",
        "notes": src.get("notes")
        or (
            None
            if tracked
            else (
                "Contact timing was estimated from your movement because "
                "shuttle/racket tracking was unavailable."
            )
        ),
        "available": timestamp is not None,
    }


def _normalize_issues(
    technique: dict[str, Any] | None,
    contact: dict[str, Any],
    phases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not technique:
        return []
    phase_seek = {
        p["id"]: p["seek_timestamp"] for p in phases if p.get("seek_timestamp") is not None
    }
    out: list[dict[str, Any]] = []
    for raw in technique.get("issues") or []:
        code = str(raw.get("code") or raw.get("issue_code") or "")
        status = str(raw.get("status") or "")
        if not status:
            # Legacy issues without status → treat as finding.
            status = "MAJOR" if raw.get("severity") == "HIGH" else (
                "MODERATE" if raw.get("severity") == "MEDIUM" else "MINOR"
            )
        phase = str(raw.get("phase") or "")
        seek = phase_seek.get(phase)
        if seek is None and contact.get("timestamp") is not None:
            if phase in {"ESTIMATED_CONTACT", "CONTACT"}:
                seek = contact["timestamp"]
        is_ie = status == "INSUFFICIENT_EVIDENCE"
        is_finding = status in {"MINOR", "MODERATE", "MAJOR"} or (
            not is_ie and status not in {"NO_ISSUE"}
        )
        ref = raw.get("reference_evidence") or {}
        out.append(
            {
                "code": code,
                "title": ISSUE_TITLES.get(code, code.replace("_", " ").title()),
                "phase": phase,
                "phase_label": PHASE_LABELS.get(phase, phase.replace("_", " ").title()),
                "severity": raw.get("severity"),
                "status": status,
                "confidence": float(raw.get("combined_confidence") or raw.get("confidence") or 0.0),
                "confidence_label": _band(
                    float(raw.get("combined_confidence") or raw.get("confidence") or 0.0)
                ),
                "measured_value": raw.get("measured_value"),
                "unit": raw.get("unit") or "",
                "reference_low": raw.get("reference_low")
                or (raw.get("reference_range") or {}).get("min"),
                "reference_high": raw.get("reference_high")
                or (raw.get("reference_range") or {}).get("max"),
                "reference_median": raw.get("reference_median") or ref.get("median"),
                "reference_percentile": raw.get("reference_percentile")
                or raw.get("percentile_position")
                or ref.get("percentile_position"),
                "reference_profile_id": raw.get("reference_profile_id")
                or technique.get("reference_profile_id")
                or "",
                "explanation": raw.get("description") or raw.get("status_reason") or "",
                "status_reason": raw.get("status_reason") or "",
                "seek_timestamp": seek,
                "is_finding": bool(is_finding) and not is_ie,
                "is_insufficient_evidence": is_ie,
                "measurement_confidence": raw.get("measurement_confidence"),
                "phase_confidence": raw.get("phase_confidence"),
                "video_quality_confidence": raw.get("video_quality_confidence"),
                "reference_confidence": raw.get("reference_confidence"),
                "combined_confidence": raw.get("combined_confidence"),
                "rule_version": raw.get("rule_version"),
                "severity_calibration_version": raw.get("severity_calibration_version"),
            }
        )
    return out


def _normalize_metrics(metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not metrics:
        return []
    catalog = [
        ("contact_elbow_angle_deg", "Elbow at contact", "deg"),
        ("contact_knee_angle_deg", "Knee at contact", "deg"),
        ("contact_shoulder_angle_deg", "Shoulder at contact", "deg"),
        ("knee_contribution_deg", "Knee contribution", "deg"),
        ("preparation_elbow_angle_deg", "Preparation elbow", "deg"),
        ("preparation_knee_angle_deg", "Preparation knee", "deg"),
        ("backswing_min_elbow_angle_deg", "Backswing elbow (min)", "deg"),
        ("peak_wrist_speed", "Peak wrist speed", ""),
        ("peak_elbow_omega", "Peak elbow angular velocity", ""),
        ("peak_elbow_omega_offset_frames", "Elbow peak timing offset", "frames"),
        ("follow_through_speed_ratio", "Follow-through speed retention", ""),
        ("acceleration_phase_fraction", "Acceleration phase fraction", ""),
        ("contact_wrist_y_normalized", "Contact wrist height", ""),
        ("recovery_frame_count", "Recovery frames", "frames"),
        ("recovery_duration_s", "Recovery duration", "s"),
        ("phase_confidence", "Phase confidence", ""),
    ]
    out: list[dict[str, Any]] = []
    for key, label, unit in catalog:
        if key not in metrics or metrics[key] is None:
            continue
        out.append(
            {
                "id": key,
                "label": label,
                "value": metrics[key],
                "unit": unit,
            }
        )
    return out


def _reference_comparisons(
    issues: list[dict[str, Any]], technique: dict[str, Any] | None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for issue in issues:
        if issue.get("measured_value") is None:
            continue
        if issue.get("reference_median") is None and issue.get("reference_low") is None:
            continue
        out.append(
            {
                "metric_id": issue.get("code"),
                "title": issue.get("title"),
                "unit": issue.get("unit") or "",
                "player_value": issue.get("measured_value"),
                "reference_median": issue.get("reference_median"),
                "reference_low": issue.get("reference_low"),
                "reference_high": issue.get("reference_high"),
                "percentile": issue.get("reference_percentile"),
                "reference_profile_id": issue.get("reference_profile_id")
                or (technique or {}).get("reference_profile_id"),
                "profile_match_level": (technique or {}).get("profile_match_level"),
                "group_label": "Reference group",
            }
        )
    return out


def _normalize_coaching(
    coaching: dict[str, Any] | None, issues: list[dict[str, Any]]
) -> dict[str, Any]:
    if not coaching:
        return {
            "available": False,
            "status": "UNAVAILABLE",
            "summary": None,
            "main_focus": None,
            "secondary": [],
            "strengths": [],
            "drills": [],
            "caveats": [
                "Analysis complete. Coaching guidance is temporarily unavailable."
            ],
        }
    status = str(coaching.get("status") or "ok").lower()
    available = status in {"ok", "fallback"} and bool(
        coaching.get("summary")
        or coaching.get("prioritized_issues")
        or coaching.get("drills")
    )
    prioritized = list(coaching.get("prioritized_issues") or [])
    issue_by_code = {i["code"]: i for i in issues}
    main = None
    secondary: list[dict[str, Any]] = []
    for idx, item in enumerate(prioritized):
        code = str(item.get("issue_code") or "")
        linked = issue_by_code.get(code)
        card = {
            "issue_code": code,
            "title": ISSUE_TITLES.get(code, code.replace("_", " ").title()),
            "explanation": item.get("explanation") or "",
            "priority": item.get("priority"),
            "related_metric_hints": item.get("related_metric_hints") or [],
            "evidence": _issue_evidence_snippet(linked) if linked else None,
            "seek_timestamp": linked.get("seek_timestamp") if linked else None,
        }
        if idx == 0:
            main = card
        elif len(secondary) < 3:
            secondary.append(card)

    drills = []
    for d in coaching.get("drills") or []:
        drills.append(
            {
                "title": d.get("name") or "Drill",
                "goal": d.get("description") or "",
                "instructions": d.get("description") or "",
                "repetitions": d.get("repetitions") or d.get("sets") or None,
                "related_issue": (d.get("targets_issue_codes") or [None])[0],
                "related_issues": d.get("targets_issue_codes") or [],
            }
        )

    strengths = []
    for s in coaching.get("strengths") or []:
        if not (s.get("evidence_refs") or s.get("description")):
            continue
        strengths.append(
            {
                "description": s.get("description") or "",
                "evidence_refs": s.get("evidence_refs") or [],
            }
        )

    return {
        "available": available,
        "status": status.upper() if available else "UNAVAILABLE",
        "summary": coaching.get("summary"),
        "main_focus": main,
        "secondary": secondary,
        "strengths": strengths,
        "drills": drills,
        "caveats": list(coaching.get("caveats") or []),
        "notes": coaching.get("notes"),
    }


def _issue_evidence_snippet(issue: dict[str, Any] | None) -> dict[str, Any] | None:
    if not issue:
        return None
    return {
        "measured": issue.get("measured_value"),
        "unit": issue.get("unit"),
        "reference_median": issue.get("reference_median"),
        "reference_percentile": issue.get("reference_percentile"),
        "reference_profile_id": issue.get("reference_profile_id"),
        "phase": issue.get("phase_label"),
    }


def _normalize_confidence(
    quality: dict[str, Any] | None,
    phases: dict[str, Any] | None,
    contact: dict[str, Any],
    technique: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    q = quality or {}
    overall = (
        q.get("analysis_confidence")
        or (evidence or {}).get("analysis_confidence")
        or (technique or {}).get("confidence")
        or (phases or {}).get("confidence")
        or 0.0
    )
    overall = float(overall)
    pose = float(q.get("pose_confidence") or q.get("mean_keypoint_confidence") or overall)
    video_q = float(q.get("video_quality_confidence") or q.get("quality_score") or overall)
    phase_c = float((phases or {}).get("confidence") or 0.0)
    contact_c = float(contact.get("confidence") or 0.0)
    ref_c = float((technique or {}).get("confidence") or 0.0)
    return {
        "overall": _band(overall),
        "overall_score": overall,
        "components": [
            {"id": "pose", "label": "Pose tracking", "level": _band(pose), "score": pose},
            {
                "id": "contact",
                "label": "Contact detection",
                "level": _band(contact_c),
                "score": contact_c,
            },
            {
                "id": "video",
                "label": "Video quality",
                "level": _band(video_q),
                "score": video_q,
            },
            {
                "id": "phase",
                "label": "Phase detection",
                "level": _band(phase_c),
                "score": phase_c,
            },
            {
                "id": "reference",
                "label": "Reference match",
                "level": _band(ref_c),
                "score": ref_c,
            },
        ],
        "message": _confidence_message(overall, q),
    }


def _normalize_keyframes(
    keyframes: dict[str, Any] | None, analysis_id: str
) -> list[dict[str, Any]]:
    if not keyframes:
        return []
    items = keyframes.get("keyframes") or keyframes.get("frames") or []
    out: list[dict[str, Any]] = []
    for item in items:
        path = item.get("path") or item.get("filename") or item.get("image")
        url = None
        if path:
            name = Path(str(path)).name
            url = f"/outputs/{analysis_id}_pose_keyframes/{name}"
            if not str(path).endswith(name):
                # Already a relative outputs path
                if str(path).startswith("/outputs/"):
                    url = str(path)
        out.append(
            {
                "label": item.get("label") or item.get("phase") or "Keyframe",
                "timestamp": item.get("timestamp"),
                "url": url,
            }
        )
    return out


def _limitations(
    confidence: dict[str, Any],
    contact: dict[str, Any],
    quality: dict[str, Any] | None,
) -> list[str]:
    notes: list[str] = []
    if confidence.get("overall") == "LOW":
        notes.append(
            confidence.get("message")
            or "Some findings may be less reliable for this recording."
        )
    if not contact.get("available"):
        notes.append("We could not reliably assess contact position in this video.")
    elif str(contact.get("contact_type") or "").upper() in {
        "KINEMATIC",
        "KINEMATIC_ESTIMATE",
        "ESTIMATED",
    }:
        notes.append(
            "Contact timing was estimated from your movement because "
            "shuttle/racket tracking was unavailable."
        )
    if quality and quality.get("warnings"):
        for w in quality.get("warnings") or []:
            notes.append(str(w))
    return notes


def _overlay_modes(meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    # Pre-rendered pose video — toggles are informational unless variants exist.
    modes = [
        {
            "id": "analysis",
            "label": "Analysis overlay",
            "available": True,
            "default": True,
            "toggleable": False,
            "note": "Skeleton and phase cues are baked into the analysis video.",
        }
    ]
    if meta and meta.get("variants"):
        for v in meta["variants"]:
            modes.append(
                {
                    "id": v.get("id"),
                    "label": v.get("label"),
                    "available": True,
                    "default": bool(v.get("default")),
                    "toggleable": True,
                    "url": v.get("url"),
                }
            )
    return modes


def _compatibility(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[bool, str]:
    from app.schemas.stroke_types import normalize_stroke_type

    try:
        ls = normalize_stroke_type(left.get("stroke_type_raw"))
        rs = normalize_stroke_type(right.get("stroke_type_raw"))
    except Exception:
        return False, "Unknown stroke types cannot be compared."
    if ls != rs:
        return False, "Different stroke types cannot be compared."
    lh = str(left.get("handedness_raw") or "").upper()
    rh = str(right.get("handedness_raw") or "").upper()
    if lh and rh and lh != rh:
        return False, "Different handedness cannot be compared."
    return True, "Compatible for metric comparison."


def _comparison_rows(
    left: dict[str, Any], right: dict[str, Any]
) -> list[dict[str, Any]]:
    keys = [
        ("contact_elbow_angle_deg", "Elbow contact", "deg"),
        ("peak_wrist_speed", "Peak wrist speed", ""),
        ("knee_contribution_deg", "Knee contribution", "deg"),
        ("phase_confidence", "Phase confidence", ""),
    ]
    left_map = {m["id"]: m["value"] for m in left.get("metrics") or []}
    right_map = {m["id"]: m["value"] for m in right.get("metrics") or []}
    # Percentiles from primary elbow issue if present.
    left_pct = _issue_percentile(left, "INSUFFICIENT_ELBOW_EXTENSION")
    right_pct = _issue_percentile(right, "INSUFFICIENT_ELBOW_EXTENSION")
    rows: list[dict[str, Any]] = []
    for key, label, unit in keys:
        lv = left_map.get(key)
        rv = right_map.get(key)
        if lv is None and rv is None:
            continue
        rows.append(
            {
                "label": label,
                "unit": unit,
                "left": lv,
                "right": rv,
                "change": _change_label(lv, rv, higher_is_better=key != "contact_wrist_y_normalized"),
            }
        )
    if left_pct is not None or right_pct is not None:
        rows.insert(
            1,
            {
                "label": "Reference percentile (elbow)",
                "unit": "",
                "left": left_pct,
                "right": right_pct,
                "change": _change_label(left_pct, right_pct, higher_is_better=True),
            },
        )
    # Contact confidence
    lc = (left.get("contact") or {}).get("confidence")
    rc = (right.get("contact") or {}).get("confidence")
    if lc is not None or rc is not None:
        rows.append(
            {
                "label": "Contact confidence",
                "unit": "",
                "left": lc,
                "right": rc,
                "change": _change_label(lc, rc, higher_is_better=True),
            }
        )
    return rows


def _issue_percentile(result: dict[str, Any], code: str) -> float | None:
    for issue in result.get("issues") or []:
        if issue.get("code") == code and issue.get("reference_percentile") is not None:
            return float(issue["reference_percentile"])
    return None


def _change_label(
    left: float | None, right: float | None, *, higher_is_better: bool
) -> str:
    if left is None or right is None:
        return "Changed"
    delta = float(right) - float(left)
    if abs(delta) < 1e-6 or abs(delta) / max(abs(float(left)), 1e-6) < 0.03:
        return "Similar"
    improved = delta > 0 if higher_is_better else delta < 0
    return "Improved" if improved else "Changed"


def _coaching_status(coaching: dict[str, Any] | None) -> str:
    if not coaching:
        return "UNAVAILABLE"
    status = str(coaching.get("status") or "").lower()
    if status == "ok":
        return "COMPLETE"
    if status in {"fallback", "skipped", "error"}:
        return status.upper()
    return "COMPLETE" if coaching.get("summary") else "UNAVAILABLE"


def _mesh_status(root: Path, analysis_id: str) -> str:
    if (root / f"{analysis_id}_mesh.mp4").is_file():
        return "COMPLETE"
    status_file = root / f"{analysis_id}_mesh.status.json"
    if status_file.is_file():
        data = _read_json(status_file) or {}
        return str(data.get("status") or "PENDING").upper()
    return "NONE"


def _band(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.5:
        return "MODERATE"
    return "LOW"


def _confidence_message(overall: float, quality: dict[str, Any]) -> str | None:
    if overall >= 0.75:
        return None
    reasons = quality.get("limitation_notes") or quality.get("warnings") or []
    if reasons:
        return str(reasons[0])
    if overall < 0.5:
        return (
            "Some findings may be less reliable because tracking confidence "
            "was limited near contact."
        )
    return "Analysis confidence is moderate — review findings with care."


def _stroke_label(raw: str | None) -> str:
    from app.schemas.stroke_types import stroke_type_label

    return stroke_type_label(raw)


def _hand_label(raw: str | None) -> str | None:
    if raw is None or str(raw).strip() == "":
        return None
    key = str(raw).upper()
    if key.startswith("R"):
        return "Right-handed"
    if key.startswith("L"):
        return "Left-handed"
    return key.title()


def _file_created_at(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat()
    except OSError:
        return _utc_now()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
